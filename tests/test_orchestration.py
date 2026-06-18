"""Orchestration tests for run_oracle's ablation conditions (P0-3).

Verifies — deterministically, no real LLM/exec — that the five pipeline
conditions share one execution budget, stop on a contract pass, and route the
right roles (retry vs repair vs reviewer). A scripted auto-responder drives every
role; a controllable `execute_eval` stub decides each execution's success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import multi_rag, pipeline
from agent.diagnostics import make_generic_contract_diagnostics as _make_generic_contract_diagnostics
from agent.generic_prompts import GENERIC_PROMPTS
from agent.llm import Reply, ScriptedLLM, ToolCall, Usage
from agent.multi_rag import (
    OracleConfig,
    _generic_task_context,
    _make_generic_code_validator,
    _role_prompts,
    run_oracle,
)
from agent.repair import (
    failed_import_packages as _failed_import_packages,
    make_generic_repair_validator as _make_generic_repair_validator,
)
from agent.roles import _atomic_write_text
from exec.session import RunResult, Session
from agent.loop import run_agent


class _AutoLLM:
    """One search_repo query, then submit; plain content in the synthesis phase."""

    def __init__(self) -> None:
        self.usage = Usage()
        self._queried = False
        self._submissions = 0

    def chat(self, messages, tools=None) -> Reply:
        names = [t["function"]["name"] for t in (tools or [])]
        if "search_repo" in names and not self._queried:
            self._queried = True
            return Reply("", [ToolCall("q", "search_repo", {"query": "how to evaluate here"})])
        submit = next((n for n in names if n.startswith("submit_")), None)
        if submit:
            if "review" in submit:
                content = "REVIEW_STATUS: PASS\n"
            elif submit == "submit_patch":
                self._submissions += 1
                return Reply("", [ToolCall("s", submit, {"edits": [
                    {
                        "old": "print('REPRO_RESULT')\n",
                        "new": "print('REPRO_RESULT')\n"
                               f"repair_marker = {self._submissions}\n",
                    }
                ], "rationale": "mark that repair used the existing file"})])
            else:
                self._submissions += 1
                system = str(messages[0].get("content", "")).lower()
                repair_line = (
                    f"repair_marker = {self._submissions}\n"
                    if "repair agent" in system
                    else ""
                )
                content = (
                    "output_path = 'predictions.json'\n"
                    "print('REPRO_RESULT')\n"
                    + repair_line
                )
            return Reply("", [ToolCall("s", submit, {"content": content})])
        return Reply("synthesis fallback\nREVIEW_STATUS: PASS\n")

    def complete(self, messages) -> str:
        return self.chat(messages).content


def _patch(monkeypatch) -> None:
    # run_oracle / ReproductionPipeline now live in agent.pipeline and resolve
    # ChatLLM / search_repo from that module's namespace.
    monkeypatch.setattr(pipeline, "ChatLLM", lambda *a, **k: _AutoLLM())
    monkeypatch.setattr(pipeline, "search_repo", lambda *a, **k: "Most relevant files:\n")


def _make_config(tmp_path: Path, outcomes: list[bool]) -> OracleConfig:
    """`outcomes[i]` = whether the i-th execute_eval call prints a valid result."""
    workdir = tmp_path / "ws"
    workdir.mkdir(exist_ok=True)
    state = {"i": 0}

    def execute_eval(session):
        ok = outcomes[min(state["i"], len(outcomes) - 1)]
        state["i"] += 1
        rr = RunResult(
            command="python eval.py",
            stdout=('REPRO_RESULT {"metric":"acc","actual":50.0,"num_examples":10}' if ok else ""),
            stderr=("" if ok else "boom: it failed"),
            exit_code=(0 if ok else 1),
            timed_out=False,
            duration_s=0.0,
        )
        session.transcript.append(rr)
        return rr

    def contract_passes(session):
        return any(r.ok and "REPRO_RESULT" in r.stdout for r in session.transcript)

    return OracleConfig(
        name="mock",
        task="reproduce the mock metric",
        metric="acc",
        expected=50.0,
        tolerance=1.0,
        attempt="t",
        workdir=workdir,
        artifact_dir=tmp_path / "art",
        eval_script="eval.py",
        make_session=lambda: Session(workdir),
        copy_clean_source=lambda: workdir.mkdir(exist_ok=True),
        execute_eval=execute_eval,
        validate_report=lambda s: s or "report",
        validate_review=lambda s: s or "review",
        public_contract_passes=contract_passes,
        verify_kwargs={"expected_num_examples": 10},
    )


def _result(cfg: OracleConfig) -> dict:
    return json.loads((cfg.workdir / "result.json").read_text())


def test_role_system_prompt_replaces_default_reproduction_prompt(tmp_path: Path) -> None:
    llm = ScriptedLLM([
        Reply("", [ToolCall("c1", "finish", {"summary": "done"})])
    ])
    result = run_agent(
        "private task",
        Session(tmp_path / "ws"),
        llm,
        system_prompt="You are the Navigator. Write a handoff.",
    )

    assert result.gave_final
    assert llm.calls[0][0]["content"] == "You are the Navigator. Write a handoff."
    assert "private task" not in llm.calls[0][0]["content"]


# ---------------------------------------------------------------------------

def test_atomic_write_text_replaces_complete_file_without_temp_artifacts(tmp_path):
    output = tmp_path / "eval.py"
    output.write_text("old\n")

    _atomic_write_text(output, "new complete content\n")

    assert output.read_text() == "new complete content\n"
    assert list(tmp_path.glob(".eval.py.*.tmp")) == []


def test_solo_is_one_shot(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, False, False])
    run_oracle(cfg, pipeline="solo")
    res = _result(cfg)
    assert res["eval_executions"] == 1
    assert not any(
        k.startswith(("repair_", "reviewer_")) for k in res["roles"]
    )
    assert "navigator" not in res["roles"]
    assert "critic" not in res["roles"]


def test_budget_is_shared_and_capped_at_five(tmp_path, monkeypatch):
    _patch(monkeypatch)
    # A task that never passes: every looped condition must stop at the budget.
    for pipeline in ("solo-repair", "full"):
        cfg = _make_config(tmp_path, outcomes=[False] * 10)
        run_oracle(cfg, pipeline=pipeline)
        res = _result(cfg)
        assert res["max_executions"] == 5
        assert res["eval_executions"] == 5, f"{pipeline} must consume the full budget"


def test_loop_stops_on_contract_pass(tmp_path, monkeypatch):
    _patch(monkeypatch)
    # Fail, fail, then pass → exactly 3 executions, not the full budget.
    for pipeline in ("solo-repair", "full"):
        cfg = _make_config(tmp_path, outcomes=[False, False, True, True])
        run_oracle(cfg, pipeline=pipeline)
        res = _result(cfg)
        # The loop stops as soon as the deterministic contract passes (3rd exec),
        # not at the budget. (Final verdict match also depends on the provenance
        # gate, which is covered in test_verify.py — not asserted here.)
        assert res["eval_executions"] == 3, f"{pipeline} should stop on the pass"
        assert res["public_evidence_found"] is True


def test_repair_vs_full_route_distinct_roles(tmp_path, monkeypatch):
    _patch(monkeypatch)
    outcomes = [False, False, True]

    cfg_repair = _make_config(tmp_path, outcomes=list(outcomes))
    run_oracle(cfg_repair, pipeline="solo-repair")
    roles = _result(cfg_repair)["roles"]
    assert any(k.startswith("repair_") for k in roles)       # feedback repair
    assert not any(k.startswith("reviewer_") for k in roles)  # no reviewer in solo-repair
    assert "critic" not in roles and "navigator" not in roles

    cfg_full = _make_config(tmp_path, outcomes=list(outcomes))
    run_oracle(cfg_full, pipeline="full")
    roles = _result(cfg_full)["roles"]
    assert any(k.startswith("repair_") for k in roles)
    assert any(k.startswith("reviewer_") for k in roles)      # reviewer participates
    assert "navigator" in roles and "critic" in roles


def test_unknown_pipeline_rejected(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    with pytest.raises(ValueError, match="unknown pipeline"):
        run_oracle(cfg, pipeline="turbo")


def test_role_prompts_are_always_generic(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])

    assert _role_prompts() == GENERIC_PROMPTS


def test_generic_task_context_exposes_protocol_but_not_private_target(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])

    context = _generic_task_context(cfg)

    assert cfg.task in context
    assert 'metric id must be "acc"' in context
    assert "num_examples` value must be 10" in context
    assert "REPRO_RESULT" in context
    assert str(cfg.expected) not in context
    assert str(cfg.tolerance) not in context


def test_generic_task_context_uses_v2_artifact_contract_when_provided(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of exactly 10 measured predictions."
    )
    cfg.public_execution_command = (
        "python eval.py --model-dir provisioned_models --data-dir provisioned_data"
    )

    context = _generic_task_context(cfg)

    assert "predictions.json" in context
    assert "exactly 10 measured predictions" in context
    assert cfg.public_execution_command in context
    assert "accept and honor this command's arguments" in context
    assert "REPRO_RESULT" not in context
    assert str(cfg.expected) not in context
    assert str(cfg.tolerance) not in context


def test_missing_artifact_diagnostic_includes_workspace_snapshot(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[False])
    cfg.public_result_protocol = "Write `predictions.json`: a JSON list."
    (cfg.workdir / "scores.csv").write_text("score\n1.0\n")
    diagnostics = _make_generic_contract_diagnostics(cfg)(Session(cfg.workdir))

    assert "predictions.json" in diagnostics[0]
    assert "scores.csv" in diagnostics[0]
    assert "Latest execution observation" in diagnostics[0]


def test_generic_code_validator_checks_public_interface_only(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of measured predictions."
    )
    validate = _make_generic_code_validator(cfg)

    assert validate("path = 'predictions.json'\n") == "path = 'predictions.json'\n"
    with pytest.raises(ValueError, match="public result artifact") as exc:
        validate("print('aggregate only')\n")
    assert "AutoAttack" not in str(exc.value)
    assert "tools/test.py" not in str(exc.value)


def test_generic_contract_diagnostics_report_shape_not_solution_hints(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[False])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of measured predictions."
    )
    diagnostics = _make_generic_contract_diagnostics(cfg)

    issues = diagnostics(Session(cfg.workdir))

    assert "public result artifact is missing" in issues[0]
    assert "AutoAttack" not in issues[0]
    assert "fine_label" not in issues[0]


def test_generic_contract_diagnostics_expose_own_artifact_shape_and_metric(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[False])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON object of measured predictions."
    )
    cfg.public_contract_passes = lambda _session: False
    cfg.verify_kwargs["recompute_fn"] = lambda _workdir: (12.5, 3)
    (cfg.workdir / "predictions.json").write_text(
        json.dumps({"run": {"id": [1, 2], "ood": [3]}})
    )
    diagnostics = _make_generic_contract_diagnostics(cfg)

    issue = diagnostics(Session(cfg.workdir))[0]

    assert "id: list[2]" in issue
    assert "ood: list[1]" in issue
    assert "acc=12.5 over n=3" in issue
    assert str(cfg.expected) not in issue
    assert str(cfg.tolerance) not in issue


def test_generic_repair_rejects_reentering_failed_package_initializer(tmp_path):
    workdir = tmp_path / "ws"
    session = Session(workdir)
    session.transcript.append(
        RunResult(
            command="python eval.py",
            stdout="",
            stderr=(
                'Traceback:\n  File "/workspace/library/plugins/__init__.py", line 1\n'
                "ModuleNotFoundError: No module named 'optional_dep'\n"
            ),
            exit_code=1,
            timed_out=False,
            duration_s=0.0,
        )
    )
    validate = _make_generic_repair_validator(
        lambda content: content,
        session,
        workdir,
        execution_start=0,
        current_code="from library.core.direct import Tool\n",
    )

    assert _failed_import_packages(session, workdir) == {"library.plugins"}
    with pytest.raises(ValueError, match="already proven to fail"):
        validate("from library.plugins.sibling import Tool\n")
    with pytest.raises(ValueError, match="made no code change"):
        validate("from library.core.direct import Tool\n")
    assert validate("from library.core.alternative import Tool\n") == (
        "from library.core.alternative import Tool\n"
    )


def test_generic_repair_uses_shared_full_file_path(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, True])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of measured predictions."
    )

    run_oracle(cfg, pipeline="solo-repair")

    result = _result(cfg)
    assert result["workflow_error"] is None
    assert result["repair_mode"] == "patch_first_full_file_fallback"
    assert "repair_1" in result["roles"]
    assert result["roles"]["repair_1"]["tool_counts"].get("submit_patch") == 1


def test_generic_context_and_runtime_probe_are_always_enabled(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of exactly 10 measured predictions."
    )

    run_oracle(cfg, pipeline="full")

    result = _result(cfg)
    assert result["runtime_probe_enabled"] is True
    assert result["runtime_probe_budget"] == multi_rag.MAX_RUNTIME_PROBES
    assert result["total_runtime_probes"] == 0

    navigator_messages = [
        json.loads(line)
        for line in (cfg.artifact_dir / "navigator_transcript.jsonl").read_text().splitlines()
    ]
    assert navigator_messages[0]["content"] == GENERIC_PROMPTS.navigator
    assert "predictions.json" in navigator_messages[1]["content"]
    assert "REPRO_RESULT" not in navigator_messages[1]["content"]
