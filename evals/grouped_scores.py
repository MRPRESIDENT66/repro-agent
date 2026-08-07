"""Generic nested score validation and grouped AUROC recomputation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.metrics import binary_auroc


@dataclass(frozen=True)
class GroupedScoresSpec:
    groups: tuple[str, ...]
    series: dict[str, int]
    positive_series: tuple[str, ...]
    negative_series: str
    positive_label: str
    negative_label: str


def parse_grouped_scores(output: dict[str, Any]) -> GroupedScoresSpec | None:
    if output.get("structure", "records") != "grouped_scores":
        return None
    groups = output.get("groups")
    series = output.get("series")
    positive = output.get("positive_series")
    negative = output.get("negative_series")
    if (
        not isinstance(groups, list)
        or not groups
        or not all(isinstance(value, str) for value in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("grouped_scores output requires string groups")
    if not isinstance(series, dict) or not series:
        raise ValueError("grouped_scores output requires series counts")
    counts = {str(name): int(count) for name, count in series.items()}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("grouped_scores series counts must be positive")
    if (
        not isinstance(positive, list)
        or not positive
        or not all(isinstance(value, str) for value in positive)
        or len(set(positive)) != len(positive)
    ):
        raise ValueError("grouped_scores output requires positive_series")
    if not isinstance(negative, str):
        raise ValueError("grouped_scores output requires negative_series")
    invalid_polarity = (
        bool(set(positive) - set(counts))
        or negative not in counts
        or negative in positive
    )
    if invalid_polarity:
        raise ValueError("grouped_scores polarity names must appear in series")
    return GroupedScoresSpec(
        tuple(groups),
        counts,
        tuple(positive),
        negative,
        str(output.get("positive_label", "positive-series")),
        str(output.get("negative_label", "negative-series")),
    )


def load_grouped_scores(spec: GroupedScoresSpec, path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != set(spec.groups):
        return None
    expected_names = set(spec.series)
    for group in spec.groups:
        block = data.get(group)
        if not isinstance(block, dict) or set(block) != expected_names:
            return None
        for name, count in spec.series.items():
            values = block.get(name)
            if not isinstance(values, list) or len(values) != count:
                return None
            try:
                numbers = [float(value) for value in values]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(value) for value in numbers):
                return None
    return data


def direction_diagnostics(spec: GroupedScoresSpec, path: Path) -> list[str]:
    data = load_grouped_scores(spec, path)
    if data is None:
        return []
    issues = []
    for group in spec.groups:
        negative = data[group][spec.negative_series]
        negative_mean = math.fsum(float(value) for value in negative) / len(negative)
        for name in spec.positive_series:
            values = data[group][name]
            positive_mean = math.fsum(float(value) for value in values) / len(values)
            if positive_mean <= negative_mean:
                issues.append(
                    "Semantically invalid score direction for "
                    f"{group}/{name}: the public protocol requires "
                    f"{spec.positive_label} scores HIGHER than {spec.negative_label}, "
                    "but observed means "
                    f"are positive={positive_mean:.6g}, negative={negative_mean:.6g}. "
                    "Correct polarity while preserving formula, order, and coverage."
                )
    return issues


def grouped_auroc(spec: GroupedScoresSpec, path: Path):
    data = load_grouped_scores(spec, path)
    if data is None:
        return None
    group_scores = []
    total = 0
    for group in spec.groups:
        negative = [float(value) for value in data[group][spec.negative_series]]
        scores = []
        for name in spec.positive_series:
            positive = [float(value) for value in data[group][name]]
            scores.append(binary_auroc(positive, negative) * 100.0)
            total += len(positive)
        group_scores.append(math.fsum(scores) / len(scores))
    return math.fsum(group_scores) / len(group_scores), total
