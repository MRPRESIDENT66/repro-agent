"""LLM layer — a tiny chat interface the agent depends on.

Narrow on purpose: the loop only needs ``complete(messages) -> str``, so the
whole control flow can be driven by :class:`ScriptedLLM` in tests (no API key, no
tokens, deterministic) while production uses :class:`DashScopeLLM`.

Carried over from the insight-agent project (one of the ~20% genuinely reused).
"""

from __future__ import annotations

from typing import Protocol

from agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


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

    def complete(self, messages: list[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


# Back-compat alias (was DashScope-specific).
DashScopeLLM = ChatLLM


class ScriptedLLM:
    """Deterministic stand-in that replays canned responses, in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)
