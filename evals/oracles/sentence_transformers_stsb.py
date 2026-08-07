"""Manifest-backed held-out Sentence-Transformers STS-B task."""

from pathlib import Path

from evals.manifest import make_oracle_config

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tasks"
    / "sentence_transformers_stsb.yaml"
)


def make_config(attempt: str):
    return make_oracle_config(MANIFEST, attempt)
