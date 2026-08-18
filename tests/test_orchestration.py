"""Deterministic tests for the adaptive orchestration graph.

Verifies deterministically, without real LLM calls, that the conditions share
one execution budget, stop on a contract pass, and route the right roles
(route, navigate, review, and repair). A scripted auto-responder drives every
role; a controllable `execute_eval` stub decides each execution's success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import pipeline
from agent.validation.contracts import (
    generic_task_context as _generic_task_context,
    make_generic_code_validator as _make_generic_code_validator,
    public_artifact_names,
)
from agent.validation.diagnostics import (
    make_generic_contract_diagnostics as _make_generic_contract_diagnostics,
)
from agent.orchestration.prompts import GENERIC_PROMPTS
from agent.runtime.llm import Reply, ScriptedLLM, ToolCall, Usage
from agent.pipeline import ReproductionPipeline, run_oracle
from agent.types import OracleConfig
from agent.orchestration.repair import (
    failed_import_packages as _failed_import_packages,
    make_generic_repair_validator as _make_generic_repair_validator,
)
from agent.orchestration.roles import RoleSynthesisError, atomic_write_text
from agent.runtime.runtime_probe import MAX_RUNTIME_PROBES
from exec.session import RunResult, Session


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
            if submit == "submit_route":
                context = str(messages[-1].get("content", "")).lower()
                semantic = any(
                    term in context
                    for term in ("out-of-distribution", "auroc", "energy score")
                )
                return Reply(
                    "",
                    [
                        ToolCall(
                            "route",
                            submit,
                            {
                                "route": "full" if semantic else "short",
                                "reasons": ["semantic task" if semantic else "explicit task"],
                                "risk_flags": ["score_direction"] if semantic else [],
                                "review_requirements": (
                                    ["Prove which population receives larger scores."]
                                    if semantic
                                    else []
                                ),
                            },
                        )
                    ],
                )
            if submit == "submit_review":
                content = """Source-grounded review of the complete evaluation path.
- `model`: `repo/model.py:12` defines model construction and checkpoint loading.
- `data`: `repo/data.py:30` defines the requested test split and sample order.
- `preprocessing`: `repo/preprocess.py:8` defines transforms and normalization.
- `metric`: `repo/metric.py:20` defines the metric and aggregation semantics.
The execution artifact follows these definitions and the public output contract.
REVIEW_STATUS: PASS
"""
            elif submit == "submit_handoff":
                content = (
                    "Source-grounded Navigator handoff covering the evaluation entry, "
                    "model and data assets, preprocessing, metric semantics, and open "
                    "risks. Repository evidence identifies the exact paths required "
                    "by the public task. "
                    + ("evidence " * 30)
                )
            elif submit == "submit_patch":
                self._submissions += 1
                return Reply("", [ToolCall("s", submit, {"edits": [
                    {
                        "old": "output_path = 'predictions.json'\n",
                        "new": "output_path = 'predictions.json'\n"
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
                    "import json\n"
                    "from pathlib import Path\n\n"
                    "output_path = 'predictions.json'\n"
                    "def write_predictions(values):\n"
                    "    Path(output_path).write_text(json.dumps(values))\n"
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
        predictions = workdir / "predictions.json"
        if ok:
            predictions.write_text(json.dumps([0] * 10))
        else:
            predictions.unlink(missing_ok=True)
        rr = RunResult(
            command="python eval.py",
            stdout=("evaluation complete" if ok else ""),
            stderr=("" if ok else "boom: it failed"),
            exit_code=(0 if ok else 1),
            timed_out=False,
            duration_s=0.0,
        )
        session.transcript.append(rr)
        return rr

    def recompute(path: Path):
        try:
            predictions = json.loads((path / "predictions.json").read_text())
        except (OSError, ValueError):
            return None
        return (50.0, len(predictions)) if isinstance(predictions, list) else None

    return OracleConfig(
        name="mock",
        task="reproduce the mock metric",
        metric="acc",
        expected=50.0,
        tolerance=1.0,
        attempt="t",
        expected_num_examples=10,
        recompute_fn=recompute,
        public_check_fn=lambda path: recompute(path) is not None,
        public_result_protocol=(
            "Write `predictions.json`: a JSON list of exactly 10 measured predictions."
        ),
        public_execution_command="python eval.py",
        workdir=workdir,
        artifact_dir=tmp_path / "art",
        eval_script="eval.py",
        make_session=lambda: Session(workdir),
        copy_clean_source=lambda: workdir.mkdir(exist_ok=True),
        execute_eval=execute_eval,
    )


def _result(cfg: OracleConfig) -> dict:
    return json.loads((cfg.workdir / "result.json").read_text())


# ---------------------------------------------------------------------------

def test_atomic_write_text_replaces_complete_file_without_temp_artifacts(tmp_path):
    output = tmp_path / "eval.py"
    output.write_text("old\n")

    atomic_write_text(output, "new complete content\n")

    assert output.read_text() == "new complete content\n"
    assert list(tmp_path.glob(".eval.py.*.tmp")) == []


def test_public_artifact_names_accepts_quoted_and_plain_filenames():
    protocol = "Write `predictions.json` and metrics.csv; ignore notes.md."

    assert public_artifact_names(protocol) == ["metrics.csv", "predictions.json"]


def test_budget_is_shared_and_capped_at_five(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False] * 10)
    run_oracle(cfg)
    res = _result(cfg)
    assert res["max_executions"] == 5
    assert res["eval_executions"] == 5


def test_loop_stops_on_contract_pass(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, False, True, True])
    run_oracle(cfg)
    res = _result(cfg)
    assert res["eval_executions"] == 3
    assert res["public_evidence_found"] is True


def test_langgraph_state_keeps_routing_metadata_but_not_large_artifacts(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, True])

    pipe = ReproductionPipeline(cfg).run()

    assert pipe.run_state["round"] == 1
    assert pipe.run_state["n_exec"] == 2
    assert pipe.run_state["public_contract_ok"] is True
    assert pipe.run_state["execution_owner"] == "repair_1"
    assert "last_execution_ok" not in pipe.run_state
    assert "failure_kind" not in pipe.run_state
    assert "eval_script_path" not in pipe.run_state
    assert "execution_log" not in pipe.run_state


def test_generic_task_context_exposes_artifact_contract_not_private_target(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])
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


def test_generic_code_validator_rejects_lambda_with_worker_processes(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = "Write `predictions.json`: a JSON list."
    validate = _make_generic_code_validator(cfg)
    code = """
from torch.utils.data import DataLoader
transform = lambda value: value
loader = DataLoader([], num_workers=2)
open('predictions.json', 'w').write('[]')
"""

    with pytest.raises(ValueError, match="cannot pickle local lambdas"):
        validate(code)


def test_generic_code_validator_allows_lambda_without_worker_processes(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = "Write `predictions.json`: a JSON list."
    validate = _make_generic_code_validator(cfg)
    code = """
from torch.utils.data import DataLoader
transform = lambda value: value
loader = DataLoader([], num_workers=0)
open('predictions.json', 'w').write('[]')
"""

    assert "num_workers=0" in validate(code)


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


def test_generic_contract_diagnostics_expose_shape_without_private_metric(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[False])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON object of measured predictions."
    )
    cfg.public_check_fn = lambda _workdir: False
    (cfg.workdir / "predictions.json").write_text(
        json.dumps({"run": {"id": [1, 2], "ood": [3]}})
    )
    diagnostics = _make_generic_contract_diagnostics(cfg)

    issue = diagnostics(Session(cfg.workdir))[0]

    assert "id: list[2]" in issue
    assert "ood: list[1]" in issue
    assert "acc=" not in issue
    assert str(cfg.expected) not in issue
    assert str(cfg.tolerance) not in issue


def test_generic_contract_diagnostics_prefer_public_semantic_invariant(tmp_path):
    cfg = _make_config(tmp_path, outcomes=[False])
    cfg.public_diagnostics_fn = lambda _workdir: [
        "Semantically invalid public score direction: OOD must be higher than ID."
    ]
    diagnostics = _make_generic_contract_diagnostics(cfg)

    issue = diagnostics(Session(cfg.workdir))[0]

    assert "OOD must be higher" in issue
    assert str(cfg.expected) not in issue


def test_adaptive_simple_task_skips_optional_agents(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])

    run_oracle(cfg)

    result = _result(cfg)
    assert set(result["roles"]) == {"router", "reproducer"}
    assert result["roles"]["router"]["tool_counts"] == {"submit_route": 1}
    assert result["routing"]["route"] == "short"
    assert result["routing"]["llm_route_valid"] is True
    assert result["collaboration_level"] == "short"


def test_adaptive_semantic_task_upgrades_to_full_collaboration(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.task = "Reproduce out-of-distribution AUROC with the documented score direction."

    run_oracle(cfg)

    result = _result(cfg)
    roles = result["roles"]
    assert "navigator" in roles
    assert "critic" in roles
    assert "reviewer_0" in roles
    assert "router" in roles
    assert result["collaboration_level"] == "full"
    assert not any(name.startswith("repair_") for name in roles)
    reviewer_transcript = (cfg.artifact_dir / "reviewer_0_transcript.jsonl").read_text()
    assert "Mandatory review requirements" in reviewer_transcript
    assert "which population receives larger scores" in reviewer_transcript


def test_adaptive_router_drops_prescriptive_score_sign_requirements(
    tmp_path, monkeypatch
):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.task = "Reproduce out-of-distribution AUROC with energy scores."
    calls = {"count": 0}

    def factory(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ScriptedLLM(
                [
                    Reply(
                        "",
                        [
                            ToolCall(
                                "route",
                                "submit_route",
                                {
                                    "route": "full",
                                    "reasons": ["score direction risk"],
                                    "risk_flags": ["score_direction"],
                                    "review_requirements": [
                                        "Write raw energy and apply no negation.",
                                        "Use the same polarity as repository AUROC, "
                                        "not the public direction.",
                                    ],
                                },
                            )
                        ],
                    )
                ]
            )
        return _AutoLLM()

    monkeypatch.setattr(pipeline, "ChatLLM", factory)
    monkeypatch.setattr(pipeline, "search_repo", lambda *a, **k: "Most relevant files:\n")

    run_oracle(cfg)

    requirements = _result(cfg)["routing"]["review_requirements"]
    assert any("whether ID or OOD" in item for item in requirements)
    assert all("negation" not in item for item in requirements)
    assert all("same polarity" not in item for item in requirements)


def test_adaptive_router_keeps_prepared_classification_on_short_path(
    tmp_path, monkeypatch
):
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.task = (
        "Reproduce top-1 accuracy for one cached model on the complete test split. "
        "A model card is provided, and model weights and data are pre-cached."
    )
    calls = {"count": 0}

    def factory(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ScriptedLLM(
                [
                    Reply(
                        "",
                        [
                            ToolCall(
                                "route",
                                "submit_route",
                                {
                                    "route": "full",
                                    "reasons": ["zero Python files"],
                                    "risk_flags": ["label mapping"],
                                    "review_requirements": ["Check the label mapping."],
                                },
                            )
                        ],
                    )
                ]
            )
        return _AutoLLM()

    monkeypatch.setattr(pipeline, "ChatLLM", factory)
    monkeypatch.setattr(pipeline, "search_repo", lambda *a, **k: "Most relevant files:\n")

    run_oracle(cfg)

    result = _result(cfg)
    assert set(result["roles"]) == {"router", "reproducer"}
    assert result["collaboration_level"] == "short"
    assert "prepared single-model classification" in " ".join(
        result["routing"]["reasons"]
    )


def test_adaptive_first_failure_adds_navigator_and_reviewer(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, True])

    run_oracle(cfg)

    result = _result(cfg)
    roles = result["roles"]
    assert "navigator" in roles
    assert "reviewer_0" in roles
    assert "repair_1" in roles
    assert not any(name.startswith("critic") for name in roles)
    assert result["collaboration_level"] == "assisted"


def test_adaptive_repeated_failure_escalates_to_critic(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, False, True])

    run_oracle(cfg)

    result = _result(cfg)
    roles = result["roles"]
    assert result["eval_executions"] == 3
    assert result["verdict"]["match"] is True
    assert result["collaboration_level"] == "full"
    assert "navigator" in roles
    assert "reviewer_0" in roles
    assert "repair_1" in roles
    assert "reviewer_1" in roles
    assert "critic_2" in roles
    assert "reviewer_2" in roles


def test_adaptive_router_uses_rule_fallback_on_invalid_llm_output(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, outcomes=[True])
    calls = {"count": 0}

    def factory(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ScriptedLLM([Reply("plain text instead of submit_route")])
        return _AutoLLM()

    monkeypatch.setattr(pipeline, "ChatLLM", factory)
    monkeypatch.setattr(pipeline, "search_repo", lambda *a, **k: "Most relevant files:\n")

    run_oracle(cfg)

    result = _result(cfg)
    assert result["verdict"]["match"] is True
    assert result["routing"]["llm_route_valid"] is False
    assert result["roles"]["router"]["fallback_used"] is True
    assert result["routing"]["route"] == "short"


def test_repair_loop_never_calls_private_recompute(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[False, True])
    private_calls = {"count": 0}
    original = cfg.recompute_fn

    def counted_recompute(path):
        private_calls["count"] += 1
        return original(path)

    cfg.recompute_fn = counted_recompute
    run_oracle(cfg)

    assert private_calls["count"] == 1


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

    run_oracle(cfg)

    result = _result(cfg)
    assert result["workflow_error"] is None
    assert result["repair_mode"] == "patch_first_full_file_fallback"
    assert "repair_1" in result["roles"]
    assert result["roles"]["repair_1"]["tool_counts"].get("submit_patch") == 1


def test_execution_count_survives_reviewer_exception(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.task = "Reproduce out-of-distribution AUROC with documented score direction."

    def fail_after_execution(*_args, **_kwargs):
        raise RuntimeError("review synthesis failed")

    monkeypatch.setattr(ReproductionPipeline, "_node_review", fail_after_execution)

    run_oracle(cfg)

    result = _result(cfg)
    assert result["verdict"]["actual"] == 50.0
    assert result["eval_executions"] == 1
    assert result["workflow_error"] == "RuntimeError: review synthesis failed"
    assert result["collaboration_pass"] is False
    assert result["collaboration_level"] == "full"


def test_reviewer_can_request_repair_without_private_verifier(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True, True])
    cfg.task = "Reproduce out-of-distribution AUROC with documented score direction."
    calls = {"count": 0}

    def request_one_repair(self, state):
        calls["count"] += 1
        status = "REPAIR_REQUIRED" if calls["count"] == 1 else "PASS"
        (self.workdir / "review_report.md").write_text(
            ("source-grounded review " * 20) + f"\nREVIEW_STATUS: {status}\n"
        )
        return {
            "reviewer_report_path": "review_report.md",
            "reviewed_execution_start": state["latest_execution_start"],
            "review_requires_repair": status == "REPAIR_REQUIRED",
        }

    monkeypatch.setattr(ReproductionPipeline, "_node_review", request_one_repair)

    run_oracle(cfg)

    result = _result(cfg)
    assert result["eval_executions"] == 2
    assert "repair_1" in result["roles"]


def test_valid_artifact_survives_repair_synthesis_failure(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.task = "Reproduce out-of-distribution AUROC with documented score direction."

    def request_repair(self, state):
        (self.workdir / "review_report.md").write_text(
            ("source-grounded review " * 20)
            + "\nREVIEW_STATUS: REPAIR_REQUIRED\n"
        )
        return {
            "reviewer_report_path": "review_report.md",
            "reviewed_execution_start": state["latest_execution_start"],
            "review_requires_repair": True,
        }

    original_run_role = pipeline.run_rag_role

    def fail_repair_role(**kwargs):
        if not kwargs["name"].startswith("repair_"):
            return original_run_role(**kwargs)
        role = {
            "steps": 4,
            "errors": 1,
            "format_errors": 4,
            "artifact_submitted": False,
            "usage": {
                "llm_calls": 4,
                "prompt_tokens": 100,
                "cache_hit_tokens": 0,
                "completion_tokens": 20,
                "cost_yuan": 0.5,
            },
            "peak_ctx_tokens": 100,
            "tool_counts": {"search_repo": 1},
            "command_indexes": [],
            "submission_trace": None,
            "runtime_probes": 0,
            "runtime_probe_recommended": False,
            "runtime_probe_hint": None,
            "probe_trace": None,
        }
        rag = {
            "dynamic": True,
            "trigger": kwargs["trigger"],
            "queries": ["concrete disputed API"],
            "calls": 1,
            "max_queries": kwargs["max_queries"],
            "usage": {
                "llm_calls": 1,
                "prompt_tokens": 10,
                "cache_hit_tokens": 0,
                "completion_tokens": 2,
                "cost_yuan": 0.1,
            },
            "trace": "repair_1_rag_trace.md",
        }
        raise RoleSynthesisError(kwargs["name"], role, rag)

    monkeypatch.setattr(ReproductionPipeline, "_node_review", request_repair)
    monkeypatch.setattr(pipeline, "run_rag_role", fail_repair_role)

    run_oracle(cfg)

    result = _result(cfg)
    assert result["verdict"]["match"] is True
    assert result["workflow_error"] is None
    assert result["collaboration_pass"] is True
    assert result["eval_executions"] == 1
    assert result["roles"]["repair_1"]["errors"] == 1
    assert result["total_cost_yuan"] >= 0.6
    assert "kept the last measured artifact" in result["workflow_warnings"][0]


def test_generic_context_and_runtime_probe_are_always_enabled(tmp_path, monkeypatch):
    _patch(monkeypatch)
    cfg = _make_config(tmp_path, outcomes=[True])
    cfg.public_result_protocol = (
        "Write `predictions.json`: a JSON list of exactly 10 measured predictions."
    )
    cfg.task = "Reproduce out-of-distribution AUROC with documented score direction."

    run_oracle(cfg)

    result = _result(cfg)
    assert result["runtime_probe_enabled"] is True
    assert result["runtime_probe_budget"] == MAX_RUNTIME_PROBES
    assert result["total_runtime_probes"] == 0

    navigator_messages = [
        json.loads(line)
        for line in (cfg.artifact_dir / "navigator_transcript.jsonl").read_text().splitlines()
    ]
    assert navigator_messages[0]["content"] == GENERIC_PROMPTS.navigator
    assert "predictions.json" in navigator_messages[1]["content"]
    assert "REPRO_RESULT" not in navigator_messages[1]["content"]
