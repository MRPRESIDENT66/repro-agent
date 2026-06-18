"""Backward-compatible facade for the reproduction runtime.

The orchestration moved to :mod:`agent.pipeline` (the pipeline + state machine)
and :mod:`agent.contracts` (task-context / validation plumbing). This module
re-exports the public surface so existing imports keep working, and still owns
the two standalone RAG-role wrappers that tests monkeypatch through this
namespace (``ChatLLM`` / ``search_repo`` are resolved here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.llm import ChatLLM
from agent.roles import (
    MAX_REPAIR_ROUNDS,
    RoleDeps,
    _dynamic_rag_role as _roles_dynamic_rag_role,
    _missing_path_hints,
    _search_evidence,
    _search_with_snippets as _roles_search_with_snippets,
)
from agent.runtime_probe import (
    MAX_RUNTIME_PROBES,
    MAX_RUNTIME_PROBES_PER_ROLE,
    RUNTIME_PROBE_TOOL,
)
from agent.types import OracleConfig
from retrieval.search import relevant_snippet, search_repo

# Contract/validation plumbing — re-exported with their historical underscore names.
from agent.contracts import (
    call_workspace_hook as _call_workspace_hook,
    extract_python as _extract_python,
    generic_task_context as _generic_task_context,
    make_generic_code_validator as _make_generic_code_validator,
    review_requires_repair as _review_requires_repair,
    role_prompts as _role_prompts,
    validate_report as _validate_report,
    validate_review as _validate_review,
)

# Orchestration core.
from agent.pipeline import (
    PipelinePolicy,
    ReproductionPipeline,
    _PipelineDone,
    build_run_record,
    emit_artifacts,
    provision_workspace,
    run_oracle,
)


def _dynamic_rag_role(**kwargs: Any) -> tuple[dict, dict]:
    """Standalone RAG-role entry point. Defaults its deps from this module's
    ``ChatLLM`` / ``search_repo`` so tests can monkeypatch them here."""
    deps = kwargs.pop(
        "deps",
        RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        ),
    )
    return _roles_dynamic_rag_role(**kwargs, deps=deps)


def _search_with_snippets(
    query: str,
    llm: ChatLLM,
    workdir: Path,
    *,
    context: str | None = None,
    extra_exclude: set[str] | None = None,
    max_files: int = 4,
) -> str:
    deps = RoleDeps(
        llm_factory=ChatLLM,
        search_fn=search_repo,
        snippet_fn=relevant_snippet,
    )
    return _roles_search_with_snippets(
        query,
        llm,
        workdir,
        context=context,
        extra_exclude=extra_exclude,
        max_files=max_files,
        deps=deps,
    )


__all__ = [
    "ChatLLM",
    "MAX_REPAIR_ROUNDS",
    "MAX_RUNTIME_PROBES",
    "MAX_RUNTIME_PROBES_PER_ROLE",
    "OracleConfig",
    "PipelinePolicy",
    "RUNTIME_PROBE_TOOL",
    "ReproductionPipeline",
    "build_run_record",
    "emit_artifacts",
    "provision_workspace",
    "relevant_snippet",
    "run_oracle",
    "search_repo",
]
