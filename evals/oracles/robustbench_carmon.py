"""Compatibility import for the manifest-backed RobustBench task."""

from evals.catalog import make_config as _make_config


def make_config(attempt: str):
    return _make_config("robustbench_carmon", attempt)
