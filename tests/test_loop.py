"""Agent loop + session mechanics — TEXT protocol (the `--no-fc` ablation twin).

Driven by ScriptedLLM over a real shell: no LLM cost, no torch — just proves the
ReAct-over-shell loop runs commands, recovers from errors, and that the session
persists on-disk state across steps. The native tool-call protocol (the default)
is covered by test_loop_fc.py.
"""

from __future__ import annotations

from pathlib import Path

from agent.llm import ScriptedLLM
from agent.loop import run_agent
from exec.session import Session


def _session(tmp_path: Path) -> Session:
    return Session(tmp_path / "ws", default_timeout=30)


def test_happy_path(tmp_path: Path) -> None:
    llm = ScriptedLLM(["```bash\necho hello\n```", "FINAL: done"])
    r = run_agent("echo a greeting", _session(tmp_path), llm, use_tools=False)
    assert r.gave_final and r.final_raw == "done"
    assert r.ran_eval and r.errors == 0
    assert r.steps == 2


def test_error_then_recover(tmp_path: Path) -> None:
    llm = ScriptedLLM(["```bash\nexit 7\n```", "```bash\necho ok\n```", "FINAL: 1"])
    r = run_agent("t", _session(tmp_path), llm, use_tools=False)
    assert r.errors == 1  # the exit 7 was observed, not fatal
    assert r.gave_final and r.steps == 3


def test_session_state_persists(tmp_path: Path) -> None:
    s = _session(tmp_path)
    # write a file in one step, read it in the next → state persists across commands
    llm = ScriptedLLM(
        ["```bash\necho persisted > note.txt\n```", "```bash\ncat note.txt\n```", "FINAL: 0"]
    )
    run_agent("t", s, llm, use_tools=False)
    assert "persisted" in s.read_file("note.txt")
    assert s.transcript[-1].stdout.strip() == "persisted"  # second cmd saw the file


def test_gives_up_at_budget(tmp_path: Path) -> None:
    llm = ScriptedLLM(["```bash\necho loop\n```"] * 12)
    r = run_agent("t", _session(tmp_path), llm, max_steps=4, use_tools=False)
    assert not r.gave_final and r.steps == 4


def test_unparseable_reply_is_a_format_error(tmp_path: Path) -> None:
    # the text protocol's weak spot: a reply with no parseable ```bash / FINAL
    llm = ScriptedLLM(["sure, I'll run `echo hi` for you", "```bash\necho hi\n```", "FINAL: 0"])
    r = run_agent("t", _session(tmp_path), llm, use_tools=False)
    assert r.format_errors == 1 and r.gave_final
    assert any("Reply with one" in m["content"] for m in r.transcript if m["role"] == "user")


def test_compress_shrinks_old_keeps_recent() -> None:
    from agent.loop import _compress

    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    for i in range(8):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "user", "content": "Observation:\n" + "x" * 1000})
    out = _compress(msgs, keep_recent=4, max_old=240)

    assert out[0]["content"] == "SYS" and out[1]["content"] == "TASK"  # head full
    assert out[-4:] == msgs[-4:]                                       # recent full
    old_blob = next(m["content"] for m in out[2:-4] if "Observation" in m["content"])
    assert "compressed" in old_blob and len(old_blob) < 1000          # old shrunk
    assert sum(len(m["content"]) for m in out) < sum(len(m["content"]) for m in msgs)


def test_replay_script_records_commands(tmp_path: Path) -> None:
    s = _session(tmp_path)
    llm = ScriptedLLM(["```bash\necho one\n```", "```bash\necho two\n```", "FINAL: 0"])
    run_agent("t", s, llm, use_tools=False)
    assert s.replay_script() == "echo one\necho two"


def test_agent_prompt_does_not_contain_expected(tmp_path: Path) -> None:
    llm = ScriptedLLM(["FINAL: done"])
    run_agent("blind task", _session(tmp_path), llm, use_tools=False)
    prompt = "\n".join(m["content"] for m in llm.calls[0])
    assert "92.60" not in prompt
    assert "private published value" in prompt


def test_final_text_inside_bash_does_not_end_agent(tmp_path: Path) -> None:
    llm = ScriptedLLM(["```bash\necho 'FINAL: 92.60'\n```", "FINAL: done"])
    result = run_agent("blind task", _session(tmp_path), llm, use_tools=False)
    assert result.steps == 2
    assert result.final_raw == "done"
