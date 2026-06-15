"""Blind deterministic verification: command evidence in, private claim out.

The agent never sees the expected value. It must print one machine-readable
result line from an executed command::

    REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000}

Only successful command stdout is trusted. ``FINAL`` text and assistant messages
are deliberately ignored, so guessing the published number cannot pass.
"""

from __future__ import annotations

import ast
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
# Real CALLS (checked in the AST, so the same word in a comment or a plain string
# does NOT count) that prove a model/data was actually loaded and a prediction/
# metric was actually computed.
_LOAD_CALLS = frozenset({
    "load_dataset", "DataLoader", "load_clean_dataset", "load_model",
    "from_pretrained", "create_model", "load_cifar10", "load_cifar100",
    "load_state_dict",
})
_PRED_CALLS = frozenset({
    "argmax", "topk", "softmax", "logsumexp", "predict", "roc_curve",
    "compute_all_metrics", "run_standard_evaluation", "clean_accuracy",
    "accuracy_score",
})
_SUBPROCESS_CALLS = frozenset({"run", "Popen", "call", "check_call", "check_output", "system"})


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_real_inline_eval(src: str) -> bool:
    """AST proof that THIS source really evaluates: an actual call that loads a
    model/data, an actual call that predicts/scores, and ``REPRO_RESULT`` in a
    real string literal. Comments and decoy marker strings don't qualify — they
    are not Call nodes."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    has_load = has_pred = has_result = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _LOAD_CALLS:
                has_load = True
            if name in _PRED_CALLS:
                has_pred = True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "REPRO_RESULT" in node.value:
                has_result = True
    return has_load and has_pred and has_result


def _delegates_in_script(src: str, workdir: Path) -> bool:
    """DELEGATION provenance: the emitting script actually shells out (a real
    subprocess CALL in its AST) to the repo's own eval entry (``tools/test.py`` …,
    which exists on disk) against a checkpoint, and emits ``REPRO_RESULT``. The
    correct clone-and-navigate behaviour — and unfakeable by string mentions
    alone, because a real subprocess call to the entry is required."""
    if "REPRO_RESULT" not in src or not _CKPT.search(src):
        return False
    if not any(
        any(True for _ in workdir.rglob(Path(m.group(0)).name))
        for m in _EVAL_ENTRY.finditer(src)
    ):
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.Call) and _call_name(n) in _SUBPROCESS_CALLS
        for n in ast.walk(tree)
    )


def _emitting_source(command: str, workdir: Path) -> str | None:
    """The source actually executed by the evidence-emitting command: the
    ``python -c`` inline body, or the first existing ``.py`` file the command
    runs. None if the command runs no resolvable script."""
    m = re.search(r"-c\s+(['\"])(.*?)\1", command, re.DOTALL)
    if m:
        return m.group(2)
    for tok in re.findall(r"[\w./-]+\.py", command):
        path = workdir / tok
        if path.is_file():
            return path.read_text(errors="replace")
    return None


def has_eval_provenance(
    workdir: str | Path,
    transcript: Iterable | None,
    command_index: int | None,
) -> bool:
    """Whether the matched evidence came from a real evaluation — bound to the
    SPECIFIC command that emitted it, not a scan of unrelated workspace files.

    Passes only if the source that command executed either (a) AST-provably loads
    a model/data, predicts, and emits ``REPRO_RESULT`` (inline eval), or (b) shells
    out to the repo's eval entry against a checkpoint (delegation). A bare
    ``echo``/``python -c "print(target)"``, a decoy marker file, or marker words in
    comments/strings therefore fail closed.
    """
    if transcript is None or not command_index:
        return False
    runs = list(transcript)
    if not (1 <= command_index <= len(runs)):
        return False
    workdir = Path(workdir)
    command = runs[command_index - 1].command
    src = _emitting_source(command, workdir)
    if src is None:
        return False
    return _is_real_inline_eval(src) or _delegates_in_script(src, workdir)


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
    if verdict.match and not has_eval_provenance(
        workdir, transcript, verdict.command_index
    ):
        verdict.match = False
        verdict.reason = "no_eval_provenance"
    return verdict
