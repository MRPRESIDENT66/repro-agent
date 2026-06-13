"""A repo-search tool the agent can call mid-task: BM25 recall + LLM rerank.

Deliberately no embeddings — the navigation ablation showed dense doesn't beat
BM25 here, so the integrated tool is the cheap, honest combination: BM25 casts
the net, the LLM reranker picks the true entry/config.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.llm import LLM
from retrieval.corpus import Doc, load_corpus
from retrieval.ladder import _purpose, _tok, bm25_search


def _rank_candidates(query: str, docs: list[Doc], k: int = 25) -> list[str]:
    """Combine BM25 recall with strong exact-path and exact-symbol signals."""
    bm25 = bm25_search(query, docs, k=min(max(k * 2, 25), len(docs)))
    bm25_rank = {path: rank for rank, path in enumerate(bm25)}
    q_lower = query.lower().replace("\\", "/")
    q_tokens = set(_tok(query))
    identifiers = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query)
        if token.lower() not in {"error", "file", "line", "openood", "python"}
    }
    scored: list[tuple[float, str]] = []
    for doc in docs:
        path_lower = doc.path.lower()
        basename = Path(doc.path).name.lower()
        path_tokens = set(_tok(doc.path))
        score = 1 / (1 + bm25_rank.get(doc.path, len(docs)))
        if path_lower in q_lower:
            score += 100
        if len(basename) >= 5 and basename in q_lower:
            score += 30
        score += 5 * len(q_tokens & path_tokens)
        score += 2 * sum(identifier in doc.text for identifier in identifiers)
        if score > 0:
            scored.append((score, doc.path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in scored[:k]]


def relevant_snippet(path: str | Path, query: str, max_chars: int = 3200) -> str:
    """Return query-centered source windows instead of only the file head."""
    text = Path(path).read_text(errors="replace")
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    terms = {
        token for token in _tok(query)
        if len(token) >= 3 and token not in {"file", "line", "error", "python"}
    }
    exact = query.lower().strip()
    ranked: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        score = 20 if exact and exact in lower else 0
        score += sum(1 for term in terms if term in lower)
        if score:
            ranked.append((score, index))
    if not ranked:
        return text[:max_chars]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: set[int] = set()
    for _, center in ranked:
        selected.update(range(max(0, center - 5), min(len(lines), center + 7)))
        rendered = "\n".join(lines[index] for index in sorted(selected))
        if len(rendered) >= max_chars:
            break
    sections: list[str] = []
    group: list[int] = []
    for index in sorted(selected):
        if group and index != group[-1] + 1:
            sections.append(_render_lines(lines, group))
            group = []
        group.append(index)
    if group:
        sections.append(_render_lines(lines, group))
    return "\n\n".join(sections)[:max_chars]


def _render_lines(lines: list[str], indexes: list[int]) -> str:
    start, end = indexes[0] + 1, indexes[-1] + 1
    body = "\n".join(f"{index + 1:>5}: {lines[index]}" for index in indexes)
    return f"# Lines {start}-{end}\n{body}"


def search_repo(
    query: str,
    root: str | Path,
    llm: LLM,
    k: int = 5,
    exclude_paths: set[str] | None = None,
    context: str | None = None,
) -> str:
    docs = load_corpus(root)
    if exclude_paths:
        docs = [doc for doc in docs if doc.path not in exclude_paths]
    if not docs:
        return "(no files indexed under the working directory)"
    ranking_query = query if not context else f"{query}\n{context}"
    candidates = _rank_candidates(ranking_query, docs)
    by_path = {d.path: d for d in docs}
    listing = "\n".join(f"{i}. {p}  —  {_purpose(by_path[p])}" for i, p in enumerate(candidates))
    evidence = f"\n\nCurrent error evidence:\n{context[:1800]}" if context else ""
    prompt = (
        f"Navigating a repo to: {query}{evidence}\n\nCandidate files:\n{listing}\n\n"
        f"Which {k} files are most useful to ACTUALLY do this (e.g. the entry "
        f"script you run + the specific config)? Reply with exact paths from the "
        f"list, one per line, most useful first. Prefer an exact path or traceback "
        f"file over files that merely mention it."
    )
    picked = [ln.strip().strip("`-* ") for ln in llm.complete([{"role": "user", "content": prompt}]).splitlines()]
    # Keep the strongest deterministic match even if the reranker overlooks an
    # exact traceback path, then let the LLM choose the remaining files.
    out = candidates[:1] + [p for p in picked if p in by_path and p not in candidates[:1]]
    for p in candidates:  # backfill from BM25 order if the LLM returned too few
        if p not in out:
            out.append(p)
    out = out[:k]
    return "Most relevant files:\n" + "\n".join(f"  {p}  —  {_purpose(by_path[p])}" for p in out)
