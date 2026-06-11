"""Measure repo-navigation: can each retrieval rung locate the eval entry+config?

Corpus = a cloned large repo (mmpretrain, ~1.5k files). Each query is a natural-
language reproduction need; gold = the file(s) you must land on. Metric =
recall@k (fraction of gold files in the top-k), averaged over queries — the
design's 'entry-point location success rate'.

    python -m retrieval.eval_nav            # keyword + BM25 (no API)
    python -m retrieval.eval_nav --dense    # + dense + hybrid (needs embeddings)
"""

from __future__ import annotations

import sys
from pathlib import Path

from retrieval.corpus import load_corpus
from retrieval.ladder import bm25_search, keyword_search

REPO = Path(__file__).resolve().parents[1] / "repos" / "mmpretrain"

# query -> gold files you must find to answer it (verified to exist in the repo).
QUERIES = [
    {"q": "How do I evaluate a trained ResNet-18 checkpoint on CIFAR-10?",
     "gold": ["tools/test.py", "configs/resnet/resnet18_8xb16_cifar10.py"]},
    {"q": "evaluate ResNet-50 model on CIFAR-100 test set",
     "gold": ["tools/test.py", "configs/resnet/resnet50_8xb16_cifar100.py"]},
    {"q": "config file for ResNet-34 trained on CIFAR-10",
     "gold": ["configs/resnet/resnet34_8xb16_cifar10.py"]},
    {"q": "run testing/inference with a config and a checkpoint to report accuracy",
     "gold": ["tools/test.py"]},
    {"q": "MobileNetV3 small configuration for CIFAR-10",
     "gold": ["configs/mobilenet_v3/mobilenet-v3-small_8xb16_cifar10.py"]},
]


def recall_at_k(method, docs, k: int) -> float:
    total = 0.0
    for item in QUERIES:
        top = set(method(item["q"], docs, k))
        gold = item["gold"]
        total += sum(g in top for g in gold) / len(gold)
    return total / len(QUERIES)


def main() -> None:
    docs = load_corpus(REPO)
    print(f"corpus: {len(docs)} files from {REPO.name}")

    methods = {"keyword": keyword_search, "bm25": bm25_search}
    if "--dense" in sys.argv:
        import numpy as np
        from retrieval.embeddings import DashScopeEmbedder
        from retrieval.index import get_doc_vectors
        from retrieval.ladder import dense_search, hybrid_search

        from agent.llm import ChatLLM
        from retrieval.ladder import llm_rerank

        emb = DashScopeEmbedder()
        vecs = get_doc_vectors(docs, emb, REPO.name)
        llm = ChatLLM()
        methods["dense"] = lambda q, d, k: dense_search(q, d, vecs, emb, k)
        methods["hybrid"] = lambda q, d, k: hybrid_search(q, d, vecs, emb, k)
        methods["+rerank"] = lambda q, d, k: llm_rerank(q, d, vecs, emb, llm, k)

    print(f"\n{'method':10} {'recall@5':>9} {'recall@10':>10}")
    for name, fn in methods.items():
        print(f"{name:10} {recall_at_k(fn, docs, 5):>9.0%} {recall_at_k(fn, docs, 10):>10.0%}")


if __name__ == "__main__":
    main()
