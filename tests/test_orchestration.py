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

from agent import multi_rag
from agent.llm import Reply, ToolCall, Usage
from agent.multi_rag import OracleConfig, run_oracle
from exec.session import RunResult, Session


class _AutoLLM:
    """One search_repo query, then submit; plain content in the synthesis phase."""

    def __init__(self) -> None:
        self.usage = Usage()
        self._queried = False

    def chat(self, messages, tools=None) -> Reply:
        names = [t["function"]["name"] for t in (tools or [])]
        if "search_repo" in names and not self._queried:
            self._queried = True
            return Reply("", [ToolCall("q", "search_repo", {"query": "how to evaluate here"})])
        submit = next((n for n in names if n.startswith("submit_")), None)
        if submit:
            content = "REVIEW_STATUS: PASS\n" if "review" in submit else "eval code"
            return Reply("", [ToolCall("s", submit, {"content": content})])
        return Reply("synthesis fallback\nREVIEW_STATUS: PASS\n")

    def complete(self, messages) -> str:
        return self.chat(messages).content


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(multi_rag, "ChatLLM", lambda *a, **k: _AutoLLM())
    monkeypatch.setattr(multi_rag, "search_repo", lambda *a, **k: "Most relevant files:\n")


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
        validate_code=lambda s: s or "code",
        validate_report=lambda s: s or "report",
        validate_review=lambda s: s or "review",
        public_contract_passes=contract_passes,
        public_contract_diagnostics=lambda session: (
            [] if contract_passes(session) else ["no valid REPRO_RESULT yet"]
        ),
        verify_kwargs={"expected_num_examples": 10},
        navigator_instruction="nav",
        reproducer_instruction="rep",
        critic_instruction="crit",
        reviewer_instruction="rev",
        repair_instruction="fix {round_index}",
    )


def _result(cfg: OracleConfig) -> dict:
    return json.loads((cfg.workdir / "result.json").read_text())


# ---------------------------------------------------------------------------

def test_solo_and_team_are_one_shot(tmp_path, monkeypatch):
    _patch(monkeypatch)
    for pipeline in ("solo", "team"):
        cfg = _make_config(tmp_path, outcomes=[False, False, False])
        run_oracle(cfg, pipeline=pipeline)
        res = _result(cfg)
        assert res["eval_executions"] == 1, f"{pipeline} must execute exactly once"
        assert not any(
            k.startswith(("repair_", "retry_", "reviewer_")) for k in res["roles"]
        ), f"{pipeline} must run no follow-up roles"
    # team adds the pre-execution roles, solo does not
    cfg_team = _make_config(tmp_path, outcomes=[True])
    run_oracle(cfg_team, pipeline="team")
    assert "navigator" in _result(cfg_team)["roles"]
    assert "critic" in _result(cfg_team)["roles"]


def test_budget_is_shared_and_capped_at_five(tmp_path, monkeypatch):
    _patch(monkeypatch)
    # A task that never passes: every looped condition must stop at the budget.
    for pipeline in ("solo-retry", "solo-repair", "full"):
        cfg = _make_config(tmp_path, outcomes=[False] * 10)
        run_oracle(cfg, pipeline=pipeline)
        res = _result(cfg)
        assert res["max_executions"] == 5
        assert res["eval_executions"] == 5, f"{pipeline} must consume the full budget"


def test_loop_stops_on_contract_pass(tmp_path, monkeypatch):
    _patch(monkeypatch)
    # Fail, fail, then pass → exactly 3 executions, not the full budget.
    for pipeline in ("solo-retry", "solo-repair", "full"):
        cfg = _make_config(tmp_path, outcomes=[False, False, True, True])
        run_oracle(cfg, pipeline=pipeline)
        res = _result(cfg)
        # The loop stops as soon as the deterministic contract passes (3rd exec),
        # not at the budget. (Final verdict match also depends on the provenance
        # gate, which is covered in test_verify.py — not asserted here.)
        assert res["eval_executions"] == 3, f"{pipeline} should stop on the pass"
        assert res["public_evidence_found"] is True


def test_retry_vs_repair_vs_full_route_distinct_roles(tmp_path, monkeypatch):
    _patch(monkeypatch)
    outcomes = [False, False, True]

    cfg_retry = _make_config(tmp_path, outcomes=list(outcomes))
    run_oracle(cfg_retry, pipeline="solo-retry")
    roles = _result(cfg_retry)["roles"]
    assert any(k.startswith("retry_") for k in roles)        # re-generation, no feedback
    assert not any(k.startswith("repair_") for k in roles)
    assert not any(k.startswith("reviewer_") for k in roles)
    assert "critic" not in roles and "navigator" not in roles

    cfg_repair = _make_config(tmp_path, outcomes=list(outcomes))
    run_oracle(cfg_repair, pipeline="solo-repair")
    roles = _result(cfg_repair)["roles"]
    assert any(k.startswith("repair_") for k in roles)       # feedback repair
    assert not any(k.startswith("retry_") for k in roles)
    assert not any(k.startswith("reviewer_") for k in roles)  # no reviewer in solo-repair

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
