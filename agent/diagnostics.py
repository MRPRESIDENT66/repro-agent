"""Generic public-contract diagnostics for verifier-driven repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.contracts import public_artifact_names
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
    transcript = session.transcript
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


def make_generic_contract_diagnostics(config: OracleConfig):
    """Describe public artifact failures without running the private verifier."""
    artifact_markers = public_artifact_names(config.public_result_protocol)

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
        if config.public_diagnostics_fn is not None:
            custom = config.public_diagnostics_fn(config.workdir)
            if custom:
                return custom
        if config.public_check_fn(config.workdir):
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
        observed = (
            " Observed public artifact evidence: " + "; ".join(observations) + "."
            if observations
            else ""
        )
        return [
            "The public result artifact exists but the deterministic verifier "
            "rejected it as malformed or incomplete. "
            "Inspect the public result protocol, repository source, and execution log."
            + observed
        ]

    return diagnostics
