#!/usr/bin/env python3
"""Prepare pinned public inputs, private gold, and model assets for STS-B."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sentence_transformers_stsb"
PUBLIC_PAIRS = DATA_DIR / "test_pairs.jsonl"
GOLD_SCORES = ROOT / "evals" / "oracles" / "gold" / "stsb_test_scores.json"

DATASET_ID = "sentence-transformers/stsb"
DATASET_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def main() -> int:
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="test",
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PUBLIC_PAIRS.open("w") as handle:
        for index, row in enumerate(dataset):
            record = {
                "id": index,
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    GOLD_SCORES.parent.mkdir(parents=True, exist_ok=True)
    GOLD_SCORES.write_text(json.dumps(list(dataset["score"]), indent=2) + "\n")

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
    )
    print(f"public pairs: {PUBLIC_PAIRS} ({len(dataset)} rows)")
    print(f"private gold: {GOLD_SCORES} ({len(dataset)} scores)")
    print(f"model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
