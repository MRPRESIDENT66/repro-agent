"""LLM layer — a tiny chat interface the agent depends on.

Two call styles, narrow on purpose:

- ``complete(messages) -> str`` — plain text in/out; what retrieval/rerank needs.
- ``chat(messages, tools=None) -> Reply`` — full fidelity: native tool calls
  (OpenAI function calling) + per-call token usage; what the agent loop needs.

Every real call accumulates into ``self.usage`` (tokens + yuan), so a run's cost
is a delta of two snapshots. :class:`ScriptedLLM` drives the whole control flow
in tests (no API key, no tokens, deterministic) — its scripted responses may be
plain strings or :class:`Reply` objects carrying tool calls.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    PRICE_INPUT_HIT,
    PRICE_INPUT_MISS,
    PRICE_OUTPUT,
)

# OpenAI wire format; assistant messages may carry "tool_calls", tool results
# carry "tool_call_id". Kept as plain dicts so transcripts serialize as-is.
Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    """Accumulated token usage. Prices are yuan per 1M tokens (agent.config)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    calls: int = 0

    def add_raw(self, u: Any) -> None:
        """Fold in one API response's ``usage`` object (provider-tolerant)."""
        if u is None:
            return
        self.calls += 1
        self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        # DeepSeek reports cache hits at the top level; vanilla OpenAI nests
        # them under prompt_tokens_details.cached_tokens.
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        if hit is None:
            details = getattr(u, "prompt_tokens_details", None)
            hit = getattr(details, "cached_tokens", 0) if details else 0
        self.cache_hit_tokens += hit or 0

    def since(self, start: "Usage | None") -> "Usage":
        """This usage minus an earlier snapshot (None = a copy of the totals)."""
        if start is None:
            return Usage(self.prompt_tokens, self.completion_tokens,
                         self.cache_hit_tokens, self.calls)
        return Usage(
            self.prompt_tokens - start.prompt_tokens,
            self.completion_tokens - start.completion_tokens,
            self.cache_hit_tokens - start.cache_hit_tokens,
            self.calls - start.calls,
        )

    @property
    def cost_yuan(self) -> float:
        miss = self.prompt_tokens - self.cache_hit_tokens
        return (
            miss * PRICE_INPUT_MISS
            + self.cache_hit_tokens * PRICE_INPUT_HIT
            + self.completion_tokens * PRICE_OUTPUT
        ) / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_yuan": round(self.cost_yuan, 4),
        }


@dataclass
class Reply:
    """One assistant turn: text and/or native tool calls, plus this call's size."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0      # real tokenizer count of the context sent
    completion_tokens: int = 0


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...
    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply: ...


class ChatLLM:
    """OpenAI-compatible chat LLM (DeepSeek by default; works with any such API)."""

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        self._model = model
        self._temperature = temperature
        self.usage = Usage()
        self._usage_lock = threading.Lock()  # concurrent agents may share one client

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            # The agent protocol is deliberately sequential. Ask the provider
            # not to emit parallel tool calls; the loop still fail-closes to
            # one executed call if a provider ignores this flag.
            kwargs["parallel_tool_calls"] = False
        response = self._client.chat.completions.create(**kwargs)
        with self._usage_lock:
            self.usage.add_raw(response.usage)

        msg = response.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}  # malformed arguments → dispatch will nudge the model
            calls.append(ToolCall(tc.id, tc.function.name, args))
        u = response.usage
        return Reply(
            msg.content or "",
            calls,
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
        )

    def complete(self, messages: list[Message]) -> str:
        return self.chat(messages).content


# Back-compat alias (was DashScope-specific).
DashScopeLLM = ChatLLM


class ScriptedLLM:
    """Deterministic stand-in replaying canned responses (str or Reply), in order."""

    def __init__(self, responses: list[str | Reply]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []
        self.usage = Usage()  # stays zero: scripted runs consume no tokens

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        r = self._responses.pop(0)
        return r if isinstance(r, Reply) else Reply(r)

    def complete(self, messages: list[Message]) -> str:
        return self.chat(messages).content
