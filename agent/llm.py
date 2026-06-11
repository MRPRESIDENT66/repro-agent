"""LLM layer — a tiny chat interface the agent depends on.

Narrow on purpose: the loop only needs ``complete(messages) -> str``, so the
whole control flow can be driven by :class:`ScriptedLLM` in tests (no API key, no
tokens, deterministic) while production uses :class:`DashScopeLLM`.

Carried over from the insight-agent project (one of the ~20% genuinely reused).
"""

from __future__ import annotations

from typing import Protocol

from agent.config import CHAT_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class DashScopeLLM:
    def __init__(self, model: str = CHAT_MODEL, temperature: float = 0.0) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=model,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            temperature=temperature,
        )

    def complete(self, messages: list[Message]) -> str:
        return self._llm.invoke(messages).content


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
