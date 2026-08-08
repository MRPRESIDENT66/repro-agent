"""Public API for declarative benchmark manifests."""

from __future__ import annotations

from pathlib import Path

from agent.types import OracleConfig
from evals.manifest.runtime import ManifestRuntime, matches_glob
from evals.manifest.schema import (
    FieldSpec,
    OracleHooks,
    TaskManifest,
    load_manifest,
)
from evals.metrics import METRICS

ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "FieldSpec",
    "ManifestRuntime",
    "OracleHooks",
    "TaskManifest",
    "load_manifest",
    "make_oracle_config",
    "matches_glob",
]


def make_oracle_config(
    manifest_path: str | Path,
    attempt: str,
    *,
    root: Path = ROOT,
    hooks: OracleHooks | None = None,
) -> OracleConfig:
    """Build an ``OracleConfig`` from one validated task manifest."""
    manifest = load_manifest(manifest_path)
    hooks = hooks or OracleHooks()
    if (
        manifest.metric not in METRICS
        and manifest.metric not in {"grouped_auroc", "near_ood_auroc"}
        and hooks.verifier is None
    ):
        raise ValueError(
            f"unknown metric {manifest.metric!r}; use one of {tuple(METRICS)} "
            "or provide a verifier hook"
        )

    runtime = ManifestRuntime(manifest, root.resolve(), attempt, hooks)
    search_exclude = {
        runtime.profile.generated_script,
        manifest.output_file,
        "navigator_report.md",
        "review_report.md",
        "reproducer_public_log.txt",
    }
    return OracleConfig(
        name=manifest.name,
        task=manifest.task_description + runtime.profile.task_suffix,
        metric=manifest.metric,
        expected=manifest.expected,
        tolerance=manifest.tolerance,
        attempt=attempt,
        expected_num_examples=manifest.expected_samples,
        recompute_fn=runtime.recompute,
        public_check_fn=runtime.public_check,
        public_diagnostics_fn=(
            runtime.public_diagnostics
            if hooks.public_diagnostics is not None
            or manifest.grouped_scores is not None
            else None
        ),
        public_result_protocol=manifest.public_result_protocol,
        public_execution_command=runtime.profile.command,
        workdir=runtime.workdir,
        artifact_dir=runtime.artifact_dir,
        eval_script=runtime.profile.generated_script,
        make_session=runtime.make_session,
        copy_clean_source=runtime.provision,
        execute_eval=runtime.execute,
        session_go_offline=runtime.profile.go_offline,
        execution_backend=runtime.profile.backend,
        chance_level=manifest.chance_level,
        search_extra_exclude=search_exclude,
        assert_blind_workspace=runtime.assert_blind,
    )
