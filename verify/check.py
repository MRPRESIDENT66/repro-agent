"""Deterministic verification — the LLM extracts, code decides.

The agent reports a metric (and we capture the log line that produced it), but
whether it *matches the claim* is computed here in plain code, never by an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER = re.compile(r"[-+]?\d+\.?\d*")


@dataclass
class Verdict:
    match: bool
    expected: float
    actual: float | None
    abs_diff: float | None
    tolerance: float
    evidence_line: str | None = None

    def as_dict(self) -> dict:
        return {
            "match": self.match,
            "expected": self.expected,
            "actual": self.actual,
            "abs_diff": self.abs_diff,
            "tolerance": self.tolerance,
            "evidence_line": self.evidence_line,
        }


def extract_number(text: str) -> float | None:
    """Pull a single numeric value out of the agent's FINAL answer."""
    m = _NUMBER.search(text.replace(",", ""))
    return float(m.group()) if m else None


def find_evidence(transcript_text: str, value: float, tol: float = 0.05) -> str | None:
    """Find the log line that actually printed ~value (provenance for the claim)."""
    for line in transcript_text.splitlines():
        for n in _NUMBER.findall(line.replace(",", "")):
            try:
                if abs(float(n) - value) <= tol:
                    return line.strip()
            except ValueError:
                continue
    return None


def verify(
    actual: float | None,
    expected: float,
    tolerance: float,
    evidence_line: str | None = None,
) -> Verdict:
    if actual is None:
        return Verdict(False, expected, None, None, tolerance, evidence_line)
    diff = abs(actual - expected)
    return Verdict(diff <= tolerance, expected, actual, diff, tolerance, evidence_line)
