"""Build and save the auditable outputs of one pipeline run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from agent.roles import MAX_REPAIR_ROUNDS
from agent.runtime_probe import MAX_RUNTIME_PROBES
from agent.types import OracleConfig

HANDOFF_FILES = (
    "navigator_report.md",
    "review_report.md",
    "reproducer_public_log.txt",
)
RETRIEVAL_RANKER = "exact_path_symbol_plus_bm25_llm"


def build_run_record(
    *,
    config: OracleConfig,
    pipeline: str,
    n_exec: int,
    roles: dict,
    rag: dict,
    workflow_error: str | None,
    rag_requirement: bool,
    handoff_requirement: bool,
    collaboration_pass: bool,
    public_evidence_found: bool,
    public_contract_diagnostics: list,
    verdict: Any,
    total_commands: int,
    probe_transcript: list,
    failure_classes: list,
    routing: dict | None = None,
) -> dict:
    """Build the data written to ``result.json``."""
    total_cost = round(
        sum(role["usage"].get("cost_yuan", 0.0) for role in roles.values())
        + sum(stage["usage"].get("cost_yuan", 0.0) for stage in rag.values()),
        4,
    )
    return {
        "task": config.task,
        "pipeline": pipeline,
        "max_executions": MAX_REPAIR_ROUNDS + 1,
        "eval_executions": n_exec,
        "blind_workspace_checked": config.assert_blind_workspace is not None,
        "agents": len(roles),
        "attempt": config.attempt,
        "execution_backend": config.execution_backend,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "retrieval_ranker": RETRIEVAL_RANKER,
        "repair_mode": "patch_first_full_file_fallback",
        "workflow_error": workflow_error,
        "total_rag_calls": sum(stage["calls"] for stage in rag.values()),
        "rag_requirement_met": rag_requirement,
        "handoff_requirement_met": handoff_requirement,
        "public_evidence_found": public_evidence_found,
        "public_contract_diagnostics": public_contract_diagnostics,
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": total_commands,
        "runtime_probe_enabled": True,
        "runtime_probe_budget": MAX_RUNTIME_PROBES,
        "total_runtime_probes": len(probe_transcript),
        "failure_classes": failure_classes,
        "routing": routing,
    }


def emit_artifacts(
    workdir: Path,
    artifact_dir: Path,
    result_json: str,
    session: Any,
    probe_transcript: list,
    *,
    eval_script: str,
) -> None:
    """Save results, replay scripts, reports, generated code, and traces."""
    replay_script = session.replay_script() + "\n"
    probe_replay_script = (
        session.probe_replay_script() + "\n" if probe_transcript else None
    )
    probe_json = json.dumps(
        [
            {
                "command": run.command,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "duration_s": run.duration_s,
            }
            for run in probe_transcript
        ],
        indent=2,
    ) + "\n"

    for output_dir in (workdir, artifact_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(result_json)
        (output_dir / "commands.sh").write_text(replay_script)
        if probe_replay_script is not None:
            (output_dir / "runtime_probes.sh").write_text(probe_replay_script)
            (output_dir / "runtime_probes.json").write_text(probe_json)

        if output_dir == workdir:
            continue
        for filename in HANDOFF_FILES + (eval_script,):
            source = workdir / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
        for pattern in ("*_rag_trace.md", "*_probe_trace.md"):
            for trace in workdir.glob(pattern):
                shutil.copy2(trace, output_dir / trace.name)
