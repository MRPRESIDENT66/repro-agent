"""Sequential function-calling loop and usage accounting."""

from __future__ import annotations

import pytest

from agent.runtime.llm import Reply, ScriptedLLM, ToolCall, Usage
from agent.runtime.loop import run_agent

TOOLS = [
    {
        "type": "function",
        "function": {"name": "submit", "parameters": {"type": "object"}},
    }
]


def _run(llm, handler, completed, **kwargs):
    return run_agent(
        llm,
        system_prompt="Role prompt without private target values.",
        initial_user_message="Public task context.",
        action_nudge="Call the submit tool.",
        tool_schemas=TOOLS,
        tool_handlers={"submit": handler},
        stop_when=completed,
        **kwargs,
    )


def test_tool_call_completes_stage_and_pairs_messages() -> None:
    state = {"done": False}

    def submit(arguments):
        state["done"] = True
        return f"accepted {arguments['content']}"

    llm = ScriptedLLM([
        Reply("", [ToolCall("call_42", "submit", {"content": "handoff"})])
    ])
    result = _run(llm, submit, lambda: state["done"])

    assert result.completed and result.steps == 1
    assistant = next(message for message in result.transcript if message.get("tool_calls"))
    tool = next(message for message in result.transcript if message["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == "call_42" == tool["tool_call_id"]
    assert tool["content"] == "accepted handoff"


def test_plain_text_is_nudged_then_recovers() -> None:
    state = {"done": False}

    def submit(_arguments):
        state["done"] = True
        return "accepted"

    llm = ScriptedLLM([
        Reply("thinking instead of acting"),
        Reply("", [ToolCall("c1", "submit", {"content": "done"})]),
    ])
    result = _run(llm, submit, lambda: state["done"])

    assert result.completed and result.format_errors == 1
    assert any(
        message.get("content") == "Call the submit tool."
        for message in result.transcript
    )


def test_tool_failure_is_returned_to_model() -> None:
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "submit", {})]),
    ])

    result = _run(
        llm,
        lambda _arguments: (_ for _ in ()).throw(ValueError("content required")),
        lambda: False,
        max_steps=1,
    )

    assert not result.completed and result.format_errors == 1
    tool = next(message for message in result.transcript if message["role"] == "tool")
    assert "content required" in tool["content"]


def test_only_first_parallel_tool_call_executes() -> None:
    seen = []
    llm = ScriptedLLM([
        Reply("", [
            ToolCall("c1", "submit", {"content": "first"}),
            ToolCall("c2", "submit", {"content": "second"}),
        ])
    ])
    result = _run(
        llm,
        lambda arguments: seen.append(arguments["content"]) or "accepted",
        lambda: bool(seen),
    )

    assert seen == ["first"]
    assert result.tool_counts == {"submit": 1}
    assert result.format_errors == 1
    skipped = [message for message in result.transcript if message.get("tool_call_id") == "c2"]
    assert skipped and "Skipped" in skipped[0]["content"]


def test_stage_context_and_prompt_are_forwarded() -> None:
    state = {"done": False}
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "submit", {"content": "done"})])
    ])

    def submit(_arguments):
        state["done"] = True
        return "accepted"

    _run(llm, submit, lambda: state["done"])

    assert llm.calls[0][0]["content"] == "Role prompt without private target values."
    assert llm.calls[0][1]["content"] == "Public task context."


def test_usage_cost_math() -> None:
    from agent import config

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=2_000_000,
        cache_hit_tokens=250_000,
        calls=3,
    )
    expected = (
        750_000 * config.PRICE_INPUT_MISS
        + 250_000 * config.PRICE_INPUT_HIT
        + 2_000_000 * config.PRICE_OUTPUT
    ) / 1e6
    assert usage.cost_yuan == pytest.approx(expected)


def test_usage_since_is_a_delta() -> None:
    total = Usage(prompt_tokens=300, completion_tokens=80, cache_hit_tokens=100, calls=3)
    start = Usage(prompt_tokens=100, completion_tokens=30, cache_hit_tokens=40, calls=1)
    delta = total.since(start)
    assert (
        delta.prompt_tokens,
        delta.completion_tokens,
        delta.cache_hit_tokens,
        delta.calls,
    ) == (200, 50, 60, 2)


def test_scripted_run_reports_zero_cost() -> None:
    state = {"done": False}
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "submit", {"content": "done"})])
    ])

    def submit(_arguments):
        state["done"] = True
        return "accepted"

    result = _run(llm, submit, lambda: state["done"])
    assert result.usage["cost_yuan"] == 0 and result.usage["llm_calls"] == 0
    assert result.peak_ctx_tokens == 0
