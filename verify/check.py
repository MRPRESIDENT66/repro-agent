"""Blind deterministic verification: command evidence in, private claim out.

The agent never sees the expected value. It must print one machine-readable
result line from an executed command::

    REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000}

Only successful command stdout is trusted. ``FINAL`` text and assistant messages
are deliberately ignored, so guessing the published number cannot pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_RESULT = re.compile(r"^\s*REPRO_RESULT\s+(\{.*\})\s*$")


def _is_direct_result_echo(command: str) -> bool:
    """Reject a shell-only relay; evidence must come from the evaluation command."""
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.startswith(("echo ", "printf ")) and "REPRO_RESULT" in stripped
    return False


@dataclass
class MetricEvidence:
    metric: str
    actual: float
    num_examples: int
    evidence_line: str
    command_index: int
    target: str | None = None

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "actual": self.actual,
            "num_examples": self.num_examples,
            "target": self.target,
            "evidence_line": self.evidence_line,
            "command_index": self.command_index,
        }


@dataclass
class Verdict:
    match: bool
    expected: float
    actual: float | None
    abs_diff: float | None
    tolerance: float
    evidence_line: str | None = None
    command_index: int | None = None
    num_examples: int | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "match": self.match,
            "expected": self.expected,
            "actual": self.actual,
            "abs_diff": self.abs_diff,
            "tolerance": self.tolerance,
            "evidence_line": self.evidence_line,
            "command_index": self.command_index,
            "num_examples": self.num_examples,
            "reason": self.reason,
        }


def extract_structured_evidence(
    transcript: Iterable,
    metric: str,
    expected_num_examples: int,
    target: str | None = None,
) -> MetricEvidence | None:
    """Return the last valid structured result from successful command stdout."""
    found: MetricEvidence | None = None
    for command_index, run in enumerate(transcript, 1):
        if not run.ok or _is_direct_result_echo(run.command):
            continue
        for line in run.stdout.splitlines():
            match = _RESULT.match(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
                evidence = MetricEvidence(
                    metric=str(payload["metric"]),
                    actual=float(payload["actual"]),
                    num_examples=int(payload["num_examples"]),
                    target=str(payload["target"]) if payload.get("target") is not None else None,
                    evidence_line=line.strip(),
                    command_index=command_index,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if evidence.metric != metric or evidence.num_examples != expected_num_examples:
                continue
            if target is not None and evidence.target != target:
                continue
            found = evidence
    return found


def verify_transcript(
    transcript: Iterable,
    *,
    expected: float,
    tolerance: float,
    metric: str,
    expected_num_examples: int,
    target: str | None = None,
) -> Verdict:
    """Verify private claim against structured evidence from executed commands."""
    evidence = extract_structured_evidence(
        transcript,
        metric=metric,
        expected_num_examples=expected_num_examples,
        target=target,
    )
    if evidence is None:
        return Verdict(
            False,
            expected,
            None,
            None,
            tolerance,
            reason="no_valid_structured_evidence",
        )
    diff = abs(evidence.actual - expected)
    return Verdict(
        match=diff <= tolerance,
        expected=expected,
        actual=evidence.actual,
        abs_diff=diff,
        tolerance=tolerance,
        evidence_line=evidence.evidence_line,
        command_index=evidence.command_index,
        num_examples=evidence.num_examples,
        reason=None if diff <= tolerance else "outside_tolerance",
    )


def has_eval_provenance(workdir: str | Path, transcript: Iterable | None = None) -> bool:
    """Heuristic V1 gate: the source that emits the result also loads data and
    predicts. Checks BOTH generated ``.py`` scripts AND inline command bodies
    (e.g. ``python -c "...eval...; print('REPRO_RESULT ...')"``) — otherwise an
    inline eval, which leaves no script file, is wrongly rejected.
    """
    data_markers = ("load_dataset", "DataLoader", "datasets.", "dataset", "CIFAR", "GLUE")
    prediction_markers = ("argmax", ".max(", "topk", "predicted", "logits")

    sources = [s.read_text(errors="replace") for s in Path(workdir).rglob("*.py")]
    if transcript is not None:
        sources.extend(run.command for run in transcript)

    for source in sources:
        if (
            "REPRO_RESULT" in source
            and any(marker in source for marker in data_markers)
            and any(marker in source for marker in prediction_markers)
        ):
            return True
    return False


def verify_run(
    transcript: Iterable,
    workdir: str | Path,
    *,
    expected: float,
    tolerance: float,
    metric: str,
    expected_num_examples: int,
    target: str | None = None,
) -> Verdict:
    """Full blind V1 gate: structured command evidence plus eval-script provenance."""
    verdict = verify_transcript(
        transcript,
        expected=expected,
        tolerance=tolerance,
        metric=metric,
        expected_num_examples=expected_num_examples,
        target=target,
    )
    if verdict.match and not has_eval_provenance(workdir, transcript):
        verdict.match = False
        verdict.reason = "no_eval_provenance"
    return verdict
