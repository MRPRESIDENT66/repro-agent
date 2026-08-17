from __future__ import annotations

import sys
from types import SimpleNamespace

import agent.runtime.llm as llm_module
from agent.runtime.loop import _assistant_message


def test_deepseek_v4_disables_thinking_and_preserves_reasoning(monkeypatch) -> None:
    captured: dict = {}
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search_repo", arguments='{"query":"entry point"}'),
    )
    message = SimpleNamespace(
        content="",
        reasoning_content="inspect the repository first",
        tool_calls=[tool_call],
    )
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        prompt_cache_hit_tokens=0,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(llm_module, "LLM_THINKING", "disabled")

    llm = llm_module.ChatLLM(model="deepseek-v4-flash")
    reply = llm.chat(
        [{"role": "user", "content": "start"}],
        tools=[{"type": "function", "function": {"name": "search_repo"}}],
    )

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["parallel_tool_calls"] is False
    assert reply.reasoning_content == "inspect the repository first"
    assert _assistant_message(reply)["reasoning_content"] == reply.reasoning_content


def test_non_deepseek_model_does_not_receive_thinking_parameter(monkeypatch) -> None:
    captured: dict = {}
    message = SimpleNamespace(content="done", tool_calls=[], reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_cache_hit_tokens=0)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kwargs: client),
    )

    llm_module.ChatLLM(model="qwen-max").chat([{"role": "user", "content": "start"}])

    assert "extra_body" not in captured


def test_chat_retries_transient_provider_errors_with_exponential_backoff(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    class RateLimitError(Exception):
        status_code = 429

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RateLimitError("slow down")
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    monkeypatch.setattr(llm_module, "LLM_MAX_RETRIES", 3)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    reply = llm_module.ChatLLM().chat([{"role": "user", "content": "start"}])

    assert reply.content == "done"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_chat_does_not_retry_non_transient_provider_errors(monkeypatch) -> None:
    calls = 0

    class AuthenticationError(Exception):
        status_code = 401

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise AuthenticationError("bad key")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))

    try:
        llm_module.ChatLLM().chat([{"role": "user", "content": "start"}])
    except AuthenticationError:
        pass
    else:
        raise AssertionError("authentication errors must not be retried")
    assert calls == 1
