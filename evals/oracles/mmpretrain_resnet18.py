"""Compatibility import for the manifest-backed mmpretrain task."""

from evals.catalog import make_config as _make_config


def make_config(attempt: str):
    return _make_config("mmpretrain_resnet18", attempt)
