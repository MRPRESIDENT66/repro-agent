"""Shared configuration types for the reproduction runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class OracleConfig:
    # Identity
    name: str
    task: str
    metric: str
    expected: float
    tolerance: float
    attempt: str
    expected_num_examples: int | None
    recompute_fn: Callable[[Path], tuple[float, int] | None]
    public_result_protocol: str
    public_execution_command: str

    # Paths
    workdir: Path
    artifact_dir: Path
    eval_script: str  # file name only, e.g. "eval_ebo.py"

    # Session lifecycle
    make_session: Callable[[], Any]
    copy_clean_source: Callable[[], None]
    execute_eval: Callable[[Any], Any]
    session_go_offline: bool = False
    execution_backend: str = "unspecified"

    # Random-chance floor for a higher-is-better metric (e.g. 50.0 for binary
    # AUROC, 100/num_classes for balanced top-1 accuracy). When set, the generic
    # path emits a framework-level "below chance => inverted direction" diagnostic
    # from the verifier-recomputed value, never the hidden target.
    chance_level: float | None = None

    # File names excluded from search (oracle-generated files, e.g. "eval_ebo.py")
    search_extra_exclude: set[str] = field(default_factory=set)

    # Blind workspace check (optional)
    assert_blind_workspace: Callable[[], None] | None = None
