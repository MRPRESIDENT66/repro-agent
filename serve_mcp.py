"""MCP server — expose the repro-agent's tools to any MCP client.

This is the *distribution* layer, not new capability: it wraps three things the
project already does so an external agent (Claude Code, etc.) can call them over
stdio. Internally the agent loop uses native function calling directly; MCP is
how the same tools cross a process / product boundary.

Tools
  - verify_evidence_line : the pure, deterministic blind-protocol core — given a
        REPRO_RESULT line and a private claim, return a match verdict. No LLM.
  - navigate_repo        : BM25 + LLM-rerank repo search (the M3 finding) — find
        the eval entry/config in a large cloned repo. Needs an LLM (lazy).
  - reproduce_artifact   : the full agent→verify pipeline on a benchmark manifest.
        Slow (minutes); needs the oracle venv + an LLM key.

Run:  python serve_mcp.py            # stdio server
Register it with an MCP client (e.g. Claude Code) pointing at this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from verify.check import verify_evidence_line as _verify_line

ROOT = Path(__file__).resolve().parent

mcp = FastMCP("repro-agent")


@mcp.tool()
def verify_evidence_line(
    evidence_line: str,
    expected: float,
    tolerance: float,
    metric: str,
    num_examples: int,
    target: str | None = None,
) -> dict[str, Any]:
    """Deterministically verify one REPRO_RESULT evidence line against a private
    claim. Returns a verdict dict (match / actual / abs_diff / reason). Pure and
    offline — this is the trustworthy core of the blind protocol."""
    return _verify_line(
        evidence_line,
        expected=expected,
        tolerance=tolerance,
        metric=metric,
        expected_num_examples=num_examples,
        target=target,
    ).as_dict()


@mcp.tool()
def navigate_repo(query: str, repo_path: str, k: int = 5) -> str:
    """Find the files most relevant to a natural-language goal in a large cloned
    repo (e.g. 'evaluate resnet18 on cifar10'). BM25 recall + LLM rerank — the
    project's M3 finding that the reranker, not the embedding, is the win.
    Requires an LLM key in the environment."""
    from agent.llm import ChatLLM
    from retrieval.search import search_repo

    root = Path(repo_path).expanduser()
    if not root.exists():
        return f"ERROR: repo path not found: {repo_path}"
    return search_repo(query, root, ChatLLM(), k=k)


@mcp.tool()
def reproduce_artifact(manifest_path: str, use_tools: bool = True) -> dict[str, Any]:
    """Run the full blind reproduction pipeline on a benchmark manifest
    (relative to the repo root) and return the staged result: stages, verdict,
    steps, token/yuan usage, and the commands run. Slow — minutes — and needs
    the oracle venv plus an LLM key."""
    from run_repro import reproduce

    return reproduce(manifest_path, use_tools=use_tools)


if __name__ == "__main__":
    mcp.run()
