"""Compatibility import for the manifest-backed OpenOOD task."""

from evals.catalog import make_config as _make_config


def make_config(attempt: str):
    return _make_config("openood_ebo", attempt)
