"""Public artifact checks and private metric recomputation."""

from __future__ import annotations

import json
from pathlib import Path

from evals.grouped_scores import (
    direction_diagnostics,
    grouped_auroc,
    load_grouped_scores,
)
from evals.manifest.output import prediction_values
from evals.manifest.schema import OracleHooks, TaskManifest
from evals.metrics import METRICS


def public_check(
    manifest: TaskManifest,
    hooks: OracleHooks,
    workdir: Path,
) -> bool:
    """Validate the declared output schema without loading hidden gold."""
    if hooks.public_check is not None:
        return hooks.public_check(manifest, workdir)
    if manifest.grouped_scores is not None:
        path = workdir / manifest.output_file
        return (
            load_grouped_scores(manifest.grouped_scores, path) is not None
            and not direction_diagnostics(manifest.grouped_scores, path)
        )
    return prediction_values(manifest, workdir) is not None


def public_diagnostics(
    manifest: TaskManifest,
    hooks: OracleHooks,
    workdir: Path,
) -> list[str]:
    if hooks.public_diagnostics is not None:
        return hooks.public_diagnostics(manifest, workdir)
    if manifest.grouped_scores is None:
        return []
    return direction_diagnostics(
        manifest.grouped_scores,
        workdir / manifest.output_file,
    )


def recompute(
    manifest: TaskManifest,
    hooks: OracleHooks,
    root: Path,
    workdir: Path,
) -> tuple[float, int] | None:
    """Recompute the private metric from a complete per-sample artifact."""
    if hooks.verifier is not None:
        return hooks.verifier(manifest, workdir)
    if manifest.metric in {"grouped_auroc", "near_ood_auroc"}:
        if manifest.grouped_scores is None:
            return None
        return grouped_auroc(
            manifest.grouped_scores,
            workdir / manifest.output_file,
        )

    predictions = prediction_values(manifest, workdir)
    if predictions is None or manifest.hidden_gold is None:
        return None
    try:
        gold_path = Path(manifest.hidden_gold).expanduser()
        if not gold_path.is_absolute():
            gold_path = root / gold_path
        gold = [float(value) for value in json.loads(gold_path.read_text())]
        if manifest.gold_limit is not None:
            gold = gold[: manifest.gold_limit]
        if len(gold) != manifest.expected_samples:
            return None
        score = METRICS[manifest.metric](gold, predictions) * manifest.metric_scale
    except (OSError, TypeError, ValueError):
        return None
    return score, manifest.expected_samples
