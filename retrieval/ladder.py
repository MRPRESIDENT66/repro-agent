"""The retrieval ladder for repo navigation: grep → BM25 → +dense → +rerank.

Each rung takes a natural-language query and the corpus, returns the top-k file
paths. The point is to measure whether each added stage locates the right
entry/config file better than the last (the design's baseline ladder, so dense
retrieval is judged against real keyword/BM25 baselines, not a strawman).
"""

from __future__ import annotations

import re

import numpy as np

from retrieval.corpus import Doc

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    # split on non-alnum, camelCase, AND letter<->digit boundaries so that
    # 'resnet18'/'cifar10' tokenize to resnet,18,cifar,10 (a fair keyword baseline,
    # not a strawman that can't match split query terms).
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", text)
    return _TOKEN.findall(text.lower())


# ---- rung 1: keyword / grep ----
def keyword_search(query: str, docs: list[Doc], k: int = 5) -> list[str]:
    # Fair grep-like baseline: count DISTINCT query terms present (not raw
    # frequency, which lets a keyword-spamming file drown a path match), and
    # weight path matches heavily (the strong signal a developer greps on).
    q = set(_tok(query))
    scored = []
    for d in docs:
        content_toks = set(_tok(d.text))
        path_toks = set(_tok(d.path))
        score = len(q & content_toks) + 5 * len(q & path_toks)
        if score:
            scored.append((score, d.path))
    scored.sort(reverse=True)
    return [p for _, p in scored[:k]]


# ---- rung 2: BM25 ----
def bm25_search(query: str, docs: list[Doc], k: int = 5) -> list[str]:
    from rank_bm25 import BM25Okapi

    corpus_toks = [_tok(d.text) for d in docs]
    bm25 = BM25Okapi(corpus_toks)
    scores = bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1][:k]
    return [docs[i].path for i in order]


# ---- rung 3: dense (embeddings) ----
def dense_search(query: str, docs: list[Doc], doc_vecs: np.ndarray, embedder, k: int = 5) -> list[str]:
    qv = np.asarray(embedder.embed([query])[0])
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    sims = doc_vecs @ qv
    order = np.argsort(sims)[::-1][:k]
    return [docs[i].path for i in order]


# ---- rung 4: hybrid (BM25 + dense via reciprocal rank fusion) ----
def hybrid_search(query: str, docs: list[Doc], doc_vecs: np.ndarray, embedder, k: int = 5) -> list[str]:
    bm = bm25_search(query, docs, k=20)
    dn = dense_search(query, docs, doc_vecs, embedder, k=20)
    rr: dict[str, float] = {}
    for rank, p in enumerate(bm):
        rr[p] = rr.get(p, 0) + 1 / (60 + rank)
    for rank, p in enumerate(dn):
        rr[p] = rr.get(p, 0) + 1 / (60 + rank)
    return [p for p, _ in sorted(rr.items(), key=lambda x: -x[1])[:k]]


# ---- rung 5: LLM rerank over the hybrid candidate pool ----
def _purpose(doc: Doc) -> str:
    # path + first meaningful content line (skip license/imports) for the LLM to judge
    for line in doc.text.splitlines()[1:]:
        s = line.strip()
        if s and not s.startswith(("#", "import", "from", '"""', "'''")):
            return s[:120]
    return ""


def llm_rerank(query: str, docs: list[Doc], doc_vecs: np.ndarray, embedder, llm, k: int = 5) -> list[str]:
    bm = bm25_search(query, docs, k=25)
    dn = dense_search(query, docs, doc_vecs, embedder, k=25)
    pool = list(dict.fromkeys(bm + dn))  # union, dedup, preserve order
    by_path = {d.path: d for d in docs}
    listing = "\n".join(f"{i}. {p}  —  {_purpose(by_path[p])}" for i, p in enumerate(pool))
    prompt = (
        f"You are navigating a code repository to accomplish: {query}\n\n"
        f"Candidate files:\n{listing}\n\n"
        f"Which {k} files are most useful to ACTUALLY do this (e.g. the entry "
        f"script you run + the specific config)? Reply with their paths only, one "
        f"per line, most useful first. Use exact paths from the list."
    )
    resp = llm.complete([{"role": "user", "content": prompt}])
    picked = [ln.strip().strip("`-* ") for ln in resp.splitlines()]
    out = [p for p in picked if p in by_path]
    # backfill from the hybrid order if the LLM returned fewer than k valid paths
    for p in pool:
        if p not in out:
            out.append(p)
    return out[:k]
