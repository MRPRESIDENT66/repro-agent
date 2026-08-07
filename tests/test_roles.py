from agent.llm import Reply, ScriptedLLM, ToolCall
from agent.roles import RoleDeps, run_rag_role
from exec.session import Session


def test_synthesis_retry_drops_invalid_response_from_model_context(tmp_path):
    role_llm = ScriptedLLM(
        [Reply("", [ToolCall("q", "search_repo", {"query": "evaluation entry point"})])]
    )
    rag_llm = ScriptedLLM([])
    synthesis_llm = ScriptedLLM(
        ["Let me search for one more file.", "print('complete source')\n"]
    )
    llms = iter((role_llm, rag_llm, synthesis_llm))

    def validate(content: str) -> str:
        if not content.startswith("print("):
            raise ValueError("source must start with Python code")
        return content

    output = tmp_path / "eval.py"
    run_rag_role(
        name="reproducer",
        workdir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        session=Session(tmp_path),
        instruction="Write executable Python source.",
        context="Public task context.",
        output_path=output,
        submit_name="submit_code",
        submit_description="Submit code.",
        validator=validate,
        trigger="initial_task",
        max_steps=1,
        synthesis_instruction="Return only complete Python source.",
        synthesis_attempts=2,
        deps=RoleDeps(
            llm_factory=lambda: next(llms),
            search_fn=lambda *_args, **_kwargs: "Most relevant files:\n",
        ),
    )

    assert output.read_text() == "print('complete source')\n"
    second_request = synthesis_llm.calls[1]
    assert all(
        message.get("content") != "Let me search for one more file."
        for message in second_request
    )
