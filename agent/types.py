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

    # Paths
    workdir: Path
    artifact_dir: Path
    eval_script: str  # file name only, e.g. "eval_ebo.py"

    # Session lifecycle
    make_session: Callable[[], Any]
    session_go_offline: bool = False

    # Oracle lifecycle
    copy_clean_source: Callable[..., None] = field(default=lambda *args: None)
    execute_eval: Callable[[Any], Any] = field(default=lambda s: None)

    # Validation
    validate_report: Callable[[str], str] | None = None
    validate_review: Callable[[str], str] | None = None

    # Contract
    public_contract_passes: Callable[[Any], bool] = field(default=lambda s: True)
    # Random-chance floor for a higher-is-better metric (e.g. 50.0 for binary
    # AUROC, 100/num_classes for balanced top-1 accuracy). When set, the generic
    # path emits a framework-level "below chance => inverted direction" diagnostic
    # from the verifier-recomputed value, never the hidden target.
    chance_level: float | None = None

    # Verify kwargs (forwarded to verify_run)
    verify_kwargs: dict = field(default_factory=dict)

    # Public machine-readable artifact contract: what output the external verifier
    # accepts, and the exact command the orchestrator uses to invoke the generated
    # program. Interface information, not an oracle solution hint.
    public_result_protocol: str = ""
    public_execution_command: str = ""

    # File names excluded from search (oracle-generated files, e.g. "eval_ebo.py")
    search_extra_exclude: set[str] = field(default_factory=set)

    # Blind workspace check (optional)
    assert_blind_workspace: Callable[..., None] | None = None

    # Files copied to artifact_dir after the run
    handoff_files: tuple[str, ...] = (
        "navigator_report.md",
        "review_report.md",
        "reproducer_public_log.txt",
    )

    retrieval_ranker: str = "exact_path_symbol_plus_bm25_llm"
