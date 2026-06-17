"""Generic public-contract diagnostics for verifier-driven repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from agent.types import OracleConfig


def _clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


def workspace_artifact_snapshot(workdir: Path, limit: int = 16) -> str:
    suffixes = {".json", ".jsonl", ".csv", ".npy", ".npz", ".txt", ".log"}
    entries: list[str] = []
    for path in sorted(workdir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            rel = path.relative_to(workdir)
        except ValueError:
            rel = path
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        entries.append(f"{rel} ({size} bytes)")
        if len(entries) >= limit:
            break
    return "; ".join(entries) if entries else "no JSON/CSV/NPY/TXT artifacts found"


def latest_execution_observation(session: Any, limit: int = 1600) -> str:
    transcript = list(getattr(session, "transcript", []))
    if not transcript:
        return "no evaluation command has run"
    latest = transcript[-1]
    status = (
        f"[timed out after {latest.duration_s:.0f}s]"
        if latest.timed_out
        else f"[exit {latest.exit_code} in {latest.duration_s:.0f}s]"
    )
    parts = [status]
    if latest.stdout.strip():
        parts.append("stdout:\n" + latest.stdout)
    if latest.stderr.strip():
        parts.append("stderr:\n" + latest.stderr)
    text = "\n".join(parts)
    return _clip(text, limit)


def below_chance_diagnostic(
    actual: float, chance_level: float, metric: str = "metric"
) -> str | None:
    """Framework-level sanity check for higher-is-better metrics."""
    if actual >= chance_level:
        return None
    return (
        f"The recomputed {metric} ({actual}) is below the {chance_level} "
        f"random-chance baseline for this higher-is-better metric. A real method "
        f"scoring below chance indicates an inverted score or label/decision "
        f"direction — correct the scoring/decision polarity so the metric exceeds "
        f"chance; do not simply negate the reported number."
    )


def make_generic_contract_diagnostics(
    config: OracleConfig, pass_gate: Callable[[Any], bool] | None = None
) -> Callable[[Any], list[str]]:
    """Expose pass/fail contract feedback without oracle-specific repair hints."""
    gate = pass_gate if pass_gate is not None else config.public_contract_passes
    artifact_markers = sorted(
        set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol))
    )

    def json_shape(value: Any, depth: int = 0) -> str:
        if isinstance(value, list):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            if depth >= 2:
                return f"dict[{len(value)} keys]"
            items = list(value.items())[:12]
            body = ", ".join(
                f"{key}: {json_shape(child, depth + 1)}" for key, child in items
            )
            suffix = ", ..." if len(value) > len(items) else ""
            return "{" + body + suffix + "}"
        return type(value).__name__

    def diagnostics(session: Any) -> list[str]:
        if gate(session):
            return []
        missing = [
            marker for marker in artifact_markers if not (config.workdir / marker).is_file()
        ]
        if missing:
            return [
                "The required public result artifact is missing after execution "
                f"(missing: {missing}). Inspect the public task, result protocol, "
                "and execution log. Current workspace artifact snapshot: "
                f"{workspace_artifact_snapshot(config.workdir)}. Latest execution observation: "
                f"{latest_execution_observation(session)}"
            ]
        observations: list[str] = []
        for marker in artifact_markers:
            path = config.workdir / marker
            if path.suffix == ".json":
                try:
                    observations.append(f"{marker}: {json_shape(json.loads(path.read_text()))}")
                except (OSError, ValueError):
                    observations.append(f"{marker}: invalid JSON")
        recompute = config.verify_kwargs.get("recompute_fn")
        measured = None
        if callable(recompute):
            try:
                measured = recompute(config.workdir)
            except Exception:
                measured = None
            if (
                isinstance(measured, tuple)
                and len(measured) >= 2
                and isinstance(measured[0], (int, float))
            ):
                observations.append(
                    f"public verifier recomputed {config.metric}={measured[0]} "
                    f"over n={measured[1]} from this artifact"
                )
        observed = (
            " Observed public artifact evidence: " + "; ".join(observations) + "."
            if observations
            else ""
        )
        base = [
            "The public result artifact exists but the deterministic verifier "
            "rejected it as malformed, incomplete, or semantically invalid. "
            "Inspect the public result protocol, repository source, and execution log."
            + observed
        ]
        if (
            config.chance_level is not None
            and isinstance(measured, tuple)
            and len(measured) >= 1
            and isinstance(measured[0], (int, float))
        ):
            below = below_chance_diagnostic(measured[0], config.chance_level, config.metric)
            if below:
                base.append(below)
        return base

    return diagnostics


# Backward-compatible private names used by older tests/imports.
_workspace_artifact_snapshot = workspace_artifact_snapshot
_latest_execution_observation = latest_execution_observation
_below_chance_diagnostic = below_chance_diagnostic
_make_generic_contract_diagnostics = make_generic_contract_diagnostics
