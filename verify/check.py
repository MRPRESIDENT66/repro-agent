"""Fail-closed verification from per-sample artifacts.

The Agent can write predictions but cannot access the hidden expected metric or
gold labels. Each Oracle supplies ``recompute_fn``; the verifier calls it in its
own context and compares the freshly computed metric with the private target.
Printed aggregate values and LLM messages are never used for grading.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

RecomputeFn = Callable[[Path], tuple[float, int] | None]


@dataclass(frozen=True)
class Verdict:
    match: bool
    expected: float
    actual: float | None
    abs_diff: float | None
    tolerance: float
    num_examples: int | None
    evidence_line: str | None
    reason: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _failed(expected: float, tolerance: float, reason: str) -> Verdict:
    return Verdict(
        match=False,
        expected=expected,
        actual=None,
        abs_diff=None,
        tolerance=tolerance,
        num_examples=None,
        evidence_line=None,
        reason=reason,
    )


def verify_run(
    workdir: str | Path,
    *,
    expected: float,
    tolerance: float,
    metric: str,
    expected_num_examples: int | None,
    recompute_fn: RecomputeFn,
) -> Verdict:
    """Recompute a metric from artifacts and compare it with the hidden target."""
    try:
        result = recompute_fn(Path(workdir))
    except Exception as exc:
        return _failed(expected, tolerance, f"recompute_error:{type(exc).__name__}")

    if not isinstance(result, tuple) or len(result) != 2:
        return _failed(expected, tolerance, "no_recomputable_predictions")

    actual, num_examples = result
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or isinstance(num_examples, bool)
        or not isinstance(num_examples, int)
        or num_examples < 0
    ):
        return _failed(expected, tolerance, "invalid_recomputed_result")

    actual = float(actual)
    difference = abs(actual - expected)
    count_matches = (
        expected_num_examples is None or num_examples == expected_num_examples
    )
    match = difference <= tolerance and count_matches
    reason = None
    if not count_matches:
        reason = "count_mismatch"
    elif difference > tolerance:
        reason = "outside_tolerance"

    return Verdict(
        match=match,
        expected=expected,
        actual=actual,
        abs_diff=difference,
        tolerance=tolerance,
        num_examples=num_examples,
        evidence_line=(
            f"verifier-recomputed {metric}={actual:.6g} over n={num_examples}"
        ),
        reason=reason,
    )
