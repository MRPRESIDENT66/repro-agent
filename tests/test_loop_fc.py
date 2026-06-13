"""Agent loop — native TOOL-CALL protocol (the default transport).

Mirror of test_loop.py: same loop, function-calling transport. Also covers what
only this mode adds: assistant↔tool message pairing, compression that must not
break that pairing, and token/cost accounting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall, Usage
from agent.loop import _compress, run_agent
from exec.session import Session


def _bash(cmd: str, cid: str = "c1") -> Reply:
    return Reply("", [ToolCall(cid, "bash", {"command": cmd})])


def _finish(summary: str = "done", cid: str = "cf") -> Reply:
    return Reply("", [ToolCall(cid, "finish", {"summary": summary})])


def _session(tmp_path: Path) -> Session:
    return Session(tmp_path / "ws", default_timeout=30)


def test_fc_happy_path(tmp_path: Path) -> None:
    llm = ScriptedLLM([_bash("echo hello"), _finish()])
    r = run_agent("echo a greeting", _session(tmp_path), llm)
    assert r.gave_final and r.final_raw == "done"
    assert r.ran_eval and r.errors == 0
    assert r.steps == 2


def test_fc_transcript_pairs_tool_results(tmp_path: Path) -> None:
    llm = ScriptedLLM([_bash("echo hi", cid="call_42"), _finish()])
    r = run_agent("t", _session(tmp_path), llm)
    a = next(m for m in r.transcript if m.get("tool_calls"))
    t = next(m for m in r.transcript if m["role"] == "tool")
    assert a["tool_calls"][0]["id"] == "call_42" == t["tool_call_id"]
    assert "hi" in t["content"]


def test_fc_error_then_recover(tmp_path: Path) -> None:
    llm = ScriptedLLM([_bash("exit 7"), _bash("echo ok", cid="c2"), _finish()])
    r = run_agent("t", _session(tmp_path), llm)
    assert r.errors == 1 and r.gave_final and r.steps == 3
    repair = next(m for m in r.transcript if m["role"] == "tool")
    assert "fix it" in repair["content"]  # tier-1 repair rides in the tool result


def test_fc_nudges_on_plain_text(tmp_path: Path) -> None:
    llm = ScriptedLLM([Reply("let me think out loud instead of acting..."), _finish()])
    r = run_agent("t", _session(tmp_path), llm)
    assert r.gave_final and r.steps == 2
    assert r.format_errors == 1  # prose-instead-of-tool-call is a format error
    assert any("Call the bash tool" in m["content"]
               for m in r.transcript if m["role"] == "user")


def test_fc_accepts_final_text_without_tool_call(tmp_path: Path) -> None:
    llm = ScriptedLLM([Reply("FINAL: done")])
    r = run_agent("t", _session(tmp_path), llm)
    assert r.gave_final and r.final_raw == "done"


def test_fc_missing_command_argument_nudges(tmp_path: Path) -> None:
    llm = ScriptedLLM([Reply("", [ToolCall("c1", "bash", {})]), _finish()])
    r = run_agent("t", _session(tmp_path), llm)
    assert r.gave_final and r.errors == 0      # not a shell error...
    assert r.format_errors == 1                # ...a protocol/format error
    t = next(m for m in r.transcript if m["role"] == "tool")
    assert "command" in t["content"]


def test_fc_executes_only_first_tool_call_per_turn(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        Reply("", [
            ToolCall("c1", "bash", {"command": "echo first > first.txt"}),
            ToolCall("c2", "bash", {"command": "echo second > second.txt"}),
        ]),
        _finish(),
    ])
    r = run_agent("t", _session(tmp_path), llm)

    assert (tmp_path / "ws" / "first.txt").exists()
    assert not (tmp_path / "ws" / "second.txt").exists()
    assert r.tool_counts == {"bash": 1, "finish": 1}
    assert r.format_errors == 1
    skipped = [m for m in r.transcript if m.get("tool_call_id") == "c2"]
    assert skipped and "Skipped" in skipped[0]["content"]


def test_fc_stage_contract_stops_without_finish(tmp_path: Path) -> None:
    session = _session(tmp_path)
    llm = ScriptedLLM([
        _bash("touch handoff.md"),
        _bash("touch should_not_run"),
    ])
    r = run_agent(
        "write handoff",
        session,
        llm,
        stop_when=lambda: (session.workdir / "handoff.md").exists(),
    )

    assert r.gave_final and r.steps == 1
    assert r.final_raw == "stage contract satisfied"
    assert not (session.workdir / "should_not_run").exists()


def test_fc_custom_tool_handler_can_complete_stage(tmp_path: Path) -> None:
    session = _session(tmp_path)
    out = session.workdir / "handoff.md"
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "submit_handoff", {"content": "grounded handoff"})]),
    ])

    def submit(arguments: dict) -> str:
        out.write_text(arguments["content"])
        return "accepted"

    r = run_agent(
        "submit handoff",
        session,
        llm,
        tool_schemas=[{"type": "function", "function": {
            "name": "submit_handoff",
            "parameters": {"type": "object"},
        }}],
        tool_handlers={"submit_handoff": submit},
        stop_when=out.exists,
    )

    assert r.gave_final and out.read_text() == "grounded handoff"
    assert r.tool_counts == {"submit_handoff": 1}


def test_fc_custom_handler_can_override_builtin_search(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "search_repo", {"query": "runtime-derived query"})]),
        _finish(),
    ])
    seen = []
    r = run_agent(
        "search",
        _session(tmp_path),
        llm,
        tool_handlers={"search_repo": lambda arguments: seen.append(arguments["query"]) or "found"},
    )

    assert seen == ["runtime-derived query"]
    assert r.tool_counts == {"search_repo": 1, "finish": 1}
    assert any(m.get("content") == "found" for m in r.transcript if m["role"] == "tool")


def test_fc_accepts_stage_context_and_custom_action_nudge(tmp_path: Path) -> None:
    llm = ScriptedLLM([Reply("thinking"), _finish()])
    run_agent(
        "task",
        _session(tmp_path),
        llm,
        initial_user_message="specific execution error",
        action_nudge="use a stage tool",
    )

    assert llm.calls[0][1]["content"] == "specific execution error"
    assert any(m.get("content") == "use a stage tool" for m in llm.calls[1])


def test_fc_agent_prompt_is_blind(tmp_path: Path) -> None:
    llm = ScriptedLLM([_finish()])
    run_agent("blind task", _session(tmp_path), llm)
    prompt = "\n".join(m["content"] for m in llm.calls[0])
    assert "92.60" not in prompt
    assert "private published value" in prompt


def test_compress_preserves_tool_call_pairing() -> None:
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for i in range(8):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "bash",
                                                  "arguments": '{"command": "x"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "Observation:\n" + "x" * 1000})
    out = _compress(msgs, keep_recent=4, max_old=240)

    assert len(out) == len(msgs)  # nothing dropped → pairing intact by construction
    for old, new in zip(msgs, out):
        assert old.get("tool_calls") == new.get("tool_calls")
        assert old.get("tool_call_id") == new.get("tool_call_id")
    shrunk = [m for m in out[2:-4] if m["role"] == "tool"]
    assert shrunk and all(len(m["content"]) < 1000 for m in shrunk)


def test_usage_cost_math() -> None:
    from agent import config

    u = Usage(prompt_tokens=1_000_000, completion_tokens=2_000_000,
              cache_hit_tokens=250_000, calls=3)
    expected = (750_000 * config.PRICE_INPUT_MISS
                + 250_000 * config.PRICE_INPUT_HIT
                + 2_000_000 * config.PRICE_OUTPUT) / 1e6
    assert u.cost_yuan == pytest.approx(expected)


def test_usage_since_is_a_delta() -> None:
    total = Usage(prompt_tokens=300, completion_tokens=80, cache_hit_tokens=100, calls=3)
    start = Usage(prompt_tokens=100, completion_tokens=30, cache_hit_tokens=40, calls=1)
    d = total.since(start)
    assert (d.prompt_tokens, d.completion_tokens, d.cache_hit_tokens, d.calls) == (200, 50, 60, 2)


def test_scripted_run_reports_zero_cost(tmp_path: Path) -> None:
    llm = ScriptedLLM([_finish()])
    r = run_agent("t", _session(tmp_path), llm)
    assert r.usage["cost_yuan"] == 0 and r.usage["llm_calls"] == 0
    assert r.peak_ctx_tokens == 0
