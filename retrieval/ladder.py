"""Small BM25 helpers used by the repo-navigation search tool."""

from __future__ import annotations

import re

from retrieval.corpus import Doc

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", text)
    return _TOKEN.findall(text.lower())


def bm25_search(query: str, docs: list[Doc], k: int = 5) -> list[str]:
    from rank_bm25 import BM25Okapi

    corpus_toks = [_tok(doc.text) for doc in docs]
    bm25 = BM25Okapi(corpus_toks)
    scores = bm25.get_scores(_tok(query))
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
    return [docs[index].path for index in order]


def _purpose(doc: Doc) -> str:
    for line in doc.text.splitlines()[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "import", "from", '"""', "'''")):
            return stripped[:120]
    return ""
