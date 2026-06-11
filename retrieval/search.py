"""A repo-search tool the agent can call mid-task: BM25 recall + LLM rerank.

Deliberately no embeddings — the navigation ablation showed dense doesn't beat
BM25 here, so the integrated tool is the cheap, honest combination: BM25 casts
the net, the LLM reranker picks the true entry/config.
"""

from __future__ import annotations

from pathlib import Path

from agent.llm import LLM
from retrieval.corpus import load_corpus
from retrieval.ladder import _purpose, bm25_search


def search_repo(query: str, root: str | Path, llm: LLM, k: int = 5) -> str:
    docs = load_corpus(root)
    if not docs:
        return "(no files indexed under the working directory)"
    candidates = bm25_search(query, docs, k=25)
    by_path = {d.path: d for d in docs}
    listing = "\n".join(f"{i}. {p}  —  {_purpose(by_path[p])}" for i, p in enumerate(candidates))
    prompt = (
        f"Navigating a repo to: {query}\n\nCandidate files:\n{listing}\n\n"
        f"Which {k} files are most useful to ACTUALLY do this (e.g. the entry "
        f"script you run + the specific config)? Reply with exact paths from the "
        f"list, one per line, most useful first."
    )
    picked = [ln.strip().strip("`-* ") for ln in llm.complete([{"role": "user", "content": prompt}]).splitlines()]
    out = [p for p in picked if p in by_path]
    for p in candidates:  # backfill from BM25 order if the LLM returned too few
        if p not in out:
            out.append(p)
    out = out[:k]
    return "Most relevant files:\n" + "\n".join(f"  {p}  —  {_purpose(by_path[p])}" for p in out)
