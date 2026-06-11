"""Embed a corpus once and cache the (normalized) vectors to disk.

Embedding ~1.8k files is ~190 API calls; caching keyed by (corpus size + first/
last path) makes re-runs free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.corpus import Doc

CACHE_DIR = Path(__file__).resolve().parents[1] / "workspaces" / "emb_cache"


def get_doc_vectors(docs: list[Doc], embedder, name: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = CACHE_DIR / f"{name}.npy"
    meta_path = CACHE_DIR / f"{name}.json"
    key = {"n": len(docs), "first": docs[0].path, "last": docs[-1].path}

    if vec_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == key:
            return np.load(vec_path)

    vecs = np.asarray(embedder.embed([d.text for d in docs]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps(key))
    return vecs
