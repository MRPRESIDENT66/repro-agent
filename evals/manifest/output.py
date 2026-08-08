"""Validation and extraction of per-sample prediction artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evals.manifest.schema import FieldSpec, TaskManifest


def _valid_field(value: Any, spec: FieldSpec, expected_index: int) -> bool:
    if spec.kind == "integer":
        valid = type(value) is int
    elif spec.kind == "number":
        valid = not isinstance(value, bool) and isinstance(value, (int, float))
    elif spec.kind == "string":
        valid = isinstance(value, str)
    else:
        valid = type(value) is bool
    if not valid:
        return False
    if spec.sequence and value != expected_index:
        return False
    if spec.kind in {"integer", "number"}:
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        if spec.minimum is not None and numeric < spec.minimum:
            return False
        if spec.maximum is not None and numeric > spec.maximum:
            return False
    return True


def prediction_values(
    manifest: TaskManifest,
    workdir: Path,
) -> list[float] | None:
    """Return validated prediction values, or ``None`` for an invalid artifact."""
    path = workdir / manifest.output_file
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or len(rows) != manifest.expected_samples:
        return None

    if manifest.output_structure == "custom":
        return None
    if manifest.output_structure == "values":
        spec = manifest.output_fields[manifest.prediction_field]
        if any(
            not _valid_field(value, spec, index)
            for index, value in enumerate(rows)
        ):
            return None
        return [float(value) for value in rows]

    expected_fields = set(manifest.output_fields)
    predictions: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        fields = set(row)
        if expected_fields - fields:
            return None
        if not manifest.allow_extra_fields and fields != expected_fields:
            return None
        if any(
            not _valid_field(row[name], spec, index)
            for name, spec in manifest.output_fields.items()
        ):
            return None
        predictions.append(float(row[manifest.prediction_field]))
    return predictions
