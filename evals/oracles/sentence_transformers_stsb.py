"""Compatibility import for the manifest-backed STS-B task."""

from pathlib import Path

from evals.catalog import make_config as _make_config

MANIFEST = Path(__file__).resolve().parents[1] / "tasks" / "sentence_transformers_stsb.yaml"


def make_config(attempt: str):
    return _make_config("sentence_transformers_stsb", attempt)
