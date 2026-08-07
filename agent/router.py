"""Cheap deterministic routing for the adaptive collaboration mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from agent.types import OracleConfig


@dataclass(frozen=True)
class RouteDecision:
    use_navigator: bool
    require_semantic_audit: bool
    repository_python_files: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


_SEMANTIC_RISK_TERMS = (
    "auroc",
    "out-of-distribution",
    "ood",
    "energy score",
    "score direction",
    "aggregation",
    "adversarial",
    "autoattack",
    "robust accuracy",
    "cosine similarity",
    "spearman",
    "multiple checkpoints",
)


def route_task(config: OracleConfig, workdir: Path) -> RouteDecision:
    """Route from public task text and repository size, without an LLM call."""
    public_text = f"{config.name}\n{config.task}\n{config.public_result_protocol}".lower()
    matched_terms = [term for term in _SEMANTIC_RISK_TERMS if term in public_text]
    python_files = sum(1 for _ in workdir.rglob("*.py"))

    reasons: list[str] = []
    if python_files >= 700:
        reasons.append(f"large repository ({python_files} Python files)")
    if matched_terms:
        reasons.append("semantic risk: " + ", ".join(matched_terms[:4]))
    if not reasons:
        reasons.append("small, explicit public task")

    return RouteDecision(
        use_navigator=python_files >= 700 or bool(matched_terms),
        require_semantic_audit=python_files >= 700 or bool(matched_terms),
        repository_python_files=python_files,
        reasons=tuple(reasons),
    )
