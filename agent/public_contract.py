"""Public artifact checks that never read hidden labels or target metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def json_list_contract(
    filename: str,
    expected_count: int,
    *,
    item_type: type = int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Callable[[Path], bool]:
    """Build a strict public checker for a flat JSON prediction list."""

    def check(workdir: Path) -> bool:
        try:
            values: Any = json.loads((workdir / filename).read_text())
        except (OSError, ValueError):
            return False
        if not isinstance(values, list) or len(values) != expected_count:
            return False
        for value in values:
            if item_type is int:
                valid_type = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid_type:
                return False
            if not math.isfinite(float(value)):
                return False
            if minimum is not None and value < minimum:
                return False
            if maximum is not None and value > maximum:
                return False
        return True

    return check
