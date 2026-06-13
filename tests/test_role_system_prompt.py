"""Role-specific prompts used by collaborative agents."""

from pathlib import Path

from agent.llm import Reply, ScriptedLLM, ToolCall
from agent.loop import run_agent
from exec.session import Session


def test_role_system_prompt_replaces_reproduction_prompt(tmp_path: Path) -> None:
    llm = ScriptedLLM([Reply("", [ToolCall("c1", "finish", {"summary": "done"})])])
    result = run_agent(
        "private task",
        Session(tmp_path / "ws"),
        llm,
        system_prompt="You are the Navigator. Write a handoff.",
    )

    assert result.gave_final
    assert llm.calls[0][0]["content"] == "You are the Navigator. Write a handoff."
    assert "private task" not in llm.calls[0][0]["content"]
