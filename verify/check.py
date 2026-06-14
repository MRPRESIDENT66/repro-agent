"""Blind deterministic verification: command evidence in, private claim out.

The agent never sees the expected value. It must print one machine-readable
result line from an executed command::

    REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000}

Only successful command stdout is trusted. ``FINAL`` text and assistant messages
are deliberately ignored, so guessing the published number cannot pass.
"""

from __future__ import annotations

import json
import math
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
    num_examples: int | None
    evidence_line: str
    command_index: int
    target: str | None = None
    datasets: dict[str, int] | None = None
    run_metrics: dict[str, dict[str, float]] | None = None
    aggregation: str | None = None

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "actual": self.actual,
            "num_examples": self.num_examples,
            "target": self.target,
            "datasets": self.datasets,
            "run_metrics": self.run_metrics,
            "aggregation": self.aggregation,
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
    datasets: dict[str, int] | None = None
    run_metrics: dict[str, dict[str, float]] | None = None
    aggregation: str | None = None
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
            "datasets": self.datasets,
            "run_metrics": self.run_metrics,
            "aggregation": self.aggregation,
            "reason": self.reason,
        }


def _parse_composite_payload(payload: dict) -> tuple[
    dict[str, int] | None,
    dict[str, dict[str, float]] | None,
    str | None,
]:
    datasets_raw = payload.get("datasets")
    datasets = None
    if datasets_raw is not None:
        if not isinstance(datasets_raw, dict):
            raise ValueError("datasets must be an object")
        datasets = {str(k): int(v) for k, v in datasets_raw.items()}

    run_metrics_raw = payload.get("run_metrics")
    run_metrics = None
    if run_metrics_raw is not None:
        if not isinstance(run_metrics_raw, dict):
            raise ValueError("run_metrics must be an object")
        run_metrics = {}
        for run, values in run_metrics_raw.items():
            if not isinstance(values, dict):
                raise ValueError("each run_metrics value must be an object")
            parsed = {str(k): float(v) for k, v in values.items()}
            if not all(math.isfinite(v) for v in parsed.values()):
                raise ValueError("run metric values must be finite")
            run_metrics[str(run)] = parsed

    aggregation = (
        str(payload["aggregation"]) if payload.get("aggregation") is not None else None
    )
    return datasets, run_metrics, aggregation


def _composite_matches(
    evidence: MetricEvidence,
    *,
    expected_datasets: dict[str, int] | None,
    expected_runs: list[str] | None,
    expected_aggregation: str | None,
) -> bool:
    if expected_datasets is not None and evidence.datasets != expected_datasets:
        return False
    if expected_aggregation is not None and evidence.aggregation != expected_aggregation:
        return False
    if expected_runs is None:
        return True
    if evidence.run_metrics is None or set(evidence.run_metrics) != set(expected_runs):
        return False
    dataset_names = set(expected_datasets or {})
    if any(set(values) != dataset_names for values in evidence.run_metrics.values()):
        return False
    computed = sum(
        sum(values.values()) / len(values)
        for values in evidence.run_metrics.values()
    ) / len(evidence.run_metrics)
    # Composite metrics are commonly reported to two decimal places. Allow the
    # aggregate and its rounded components to differ by one hundredth while
    # still rejecting materially unsupported claims.
    return abs(computed - evidence.actual) <= 0.011


def extract_structured_evidence(
    transcript: Iterable,
    metric: str,
    expected_num_examples: int | None,
    target: str | None = None,
    expected_datasets: dict[str, int] | None = None,
    expected_runs: list[str] | None = None,
    expected_aggregation: str | None = None,
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
                datasets, run_metrics, aggregation = _parse_composite_payload(payload)
                evidence = MetricEvidence(
                    metric=str(payload["metric"]),
                    actual=float(payload["actual"]),
                    num_examples=(
                        int(payload["num_examples"])
                        if payload.get("num_examples") is not None
                        else None
                    ),
                    target=str(payload["target"]) if payload.get("target") is not None else None,
                    evidence_line=line.strip(),
                    command_index=command_index,
                    datasets=datasets,
                    run_metrics=run_metrics,
                    aggregation=aggregation,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if evidence.metric != metric:
                continue
            if expected_num_examples is not None and evidence.num_examples != expected_num_examples:
                continue
            if target is not None and evidence.target != target:
                continue
            if not _composite_matches(
                evidence,
                expected_datasets=expected_datasets,
                expected_runs=expected_runs,
                expected_aggregation=expected_aggregation,
            ):
                continue
            found = evidence
    return found


def verify_transcript(
    transcript: Iterable,
    *,
    expected: float,
    tolerance: float,
    metric: str,
    expected_num_examples: int | None,
    target: str | None = None,
    expected_datasets: dict[str, int] | None = None,
    expected_runs: list[str] | None = None,
    expected_aggregation: str | None = None,
) -> Verdict:
    """Verify private claim against structured evidence from executed commands."""
    evidence = extract_structured_evidence(
        transcript,
        metric=metric,
        expected_num_examples=expected_num_examples,
        target=target,
        expected_datasets=expected_datasets,
        expected_runs=expected_runs,
        expected_aggregation=expected_aggregation,
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
        datasets=evidence.datasets,
        run_metrics=evidence.run_metrics,
        aggregation=evidence.aggregation,
        reason=None if diff <= tolerance else "outside_tolerance",
    )


def verify_evidence_line(
    evidence_line: str,
    *,
    expected: float,
    tolerance: float,
    metric: str,
    expected_num_examples: int,
    target: str | None = None,
) -> Verdict:
    """Verify a single ``REPRO_RESULT`` line against a private claim — the pure,
    standalone core of the blind protocol (no transcript, no filesystem, no LLM).

    This is what the MCP ``verify_evidence_line`` tool exposes: hand it one
    structured evidence line plus what you privately expect, get a deterministic
    verdict back. Malformed / mismatched lines fail closed with a reason.
    """
    match = _RESULT.match(evidence_line.strip())
    if not match:
        return Verdict(False, expected, None, None, tolerance, reason="not_a_repro_result_line")
    try:
        payload = json.loads(match.group(1))
        actual = float(payload["actual"])
        got_metric = str(payload["metric"])
        got_n = int(payload["num_examples"])
        got_target = str(payload["target"]) if payload.get("target") is not None else None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return Verdict(False, expected, None, None, tolerance, reason="malformed_evidence")
    if got_metric != metric:
        return Verdict(False, expected, None, None, tolerance, reason="metric_mismatch")
    if got_n != expected_num_examples:
        return Verdict(False, expected, actual, None, tolerance, reason="num_examples_mismatch")
    if target is not None and got_target != target:
        return Verdict(False, expected, actual, None, tolerance, reason="target_mismatch")
    diff = abs(actual - expected)
    return Verdict(
        match=diff <= tolerance,
        expected=expected,
        actual=actual,
        abs_diff=diff,
        tolerance=tolerance,
        evidence_line=evidence_line.strip(),
        num_examples=got_n,
        reason=None if diff <= tolerance else "outside_tolerance",
    )


_EVAL_ENTRY = re.compile(r"[\w./-]*(?:test|eval|validate|val)\.py")
_CKPT = re.compile(r"[\w./-]+\.(?:pth|pt|ckpt|bin|safetensors)\b")


def _delegates_to_repo_eval(command: str, workdir: Path) -> bool:
    """Provenance via DELEGATION: the evidence-emitting command ran the cloned
    repo's own eval entry (e.g. ``tools/test.py <config> <checkpoint>``) and
    parsed its output. For a clone-and-navigate oracle this is the *correct*
    behaviour — the prediction/argmax lives in the repo's library code, not in a
    script the agent rewrote — so the inline-marker heuristic below would
    false-negative it. It still can't be faked: you can't get the real value +
    num_examples without actually running the entry against the checkpoint.
    """
    if "REPRO_RESULT" not in command or not _CKPT.search(command):
        return False
    for m in _EVAL_ENTRY.finditer(command):
        name = Path(m.group(0)).name
        if any(True for _ in Path(workdir).rglob(name)):  # the entry exists on disk
            return True
    return False


def has_eval_provenance(workdir: str | Path, transcript: Iterable | None = None) -> bool:
    """Heuristic V1 gate: the result came from a real evaluation, not an echo.

    Accepts EITHER pattern:
      1. **inline eval** — a ``.py`` script or command body that emits
         ``REPRO_RESULT`` *and* itself loads data *and* predicts (the
         library-load oracles, where the agent writes a self-contained eval).
      2. **delegation** — the emitting command invoked the cloned repo's own
         eval entry (``tools/test.py`` …) against the checkpoint (the
         clone-and-navigate oracle, where the agent correctly reuses the repo's
         harness instead of reinventing it). See :func:`_delegates_to_repo_eval`.
    """
    data_markers = ("load_dataset", "DataLoader", "datasets.", "dataset", "CIFAR", "GLUE")
    prediction_markers = (
        "argmax", ".max(", "topk", "predicted", "logits", "logsumexp",
        "roc_curve", "compute_all_metrics",
        # library-API eval calls: a real metric computed by a standard eval/attack
        # routine (e.g. robustbench/AutoAttack) rather than a hand-rolled argmax.
        "run_standard_evaluation", "clean_accuracy", "AutoAttack", "accuracy_score",
    )
    workdir = Path(workdir)

    commands = [run.command for run in transcript] if transcript is not None else []
    sources = [s.read_text(errors="replace") for s in workdir.rglob("*.py")]

    # Delegation provenance: the repo's own eval entry (tools/test.py …) is run
    # against the checkpoint. This can appear either directly on the command line
    # (single-step oracle) OR inside a wrapper ``.py`` the agent wrote that shells
    # out to the entry (the multi-agent pattern). Both are legitimate
    # clone-and-navigate behaviour and equally hard to fake — you still cannot
    # produce the real value + num_examples without actually running the entry.
    if any(_delegates_to_repo_eval(t, workdir) for t in (*commands, *sources)):
        return True

    for source in (*sources, *commands):
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
    expected_num_examples: int | None,
    target: str | None = None,
    expected_datasets: dict[str, int] | None = None,
    expected_runs: list[str] | None = None,
    expected_aggregation: str | None = None,
) -> Verdict:
    """Full blind V1 gate: structured command evidence plus eval-script provenance."""
    verdict = verify_transcript(
        transcript,
        expected=expected,
        tolerance=tolerance,
        metric=metric,
        expected_num_examples=expected_num_examples,
        target=target,
        expected_datasets=expected_datasets,
        expected_runs=expected_runs,
        expected_aggregation=expected_aggregation,
    )
    if verdict.match and not has_eval_provenance(workdir, transcript):
        verdict.match = False
        verdict.reason = "no_eval_provenance"
    return verdict
