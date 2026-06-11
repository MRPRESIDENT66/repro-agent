"""Bootstrap CI sanity (no LLM, no network)."""

from __future__ import annotations

from evals.significance import clustered_bootstrap, paired_bootstrap


def test_paired_identical_spans_zero() -> None:
    a = [True, False, True, True, False]
    r = paired_bootstrap(a, a)
    assert r["delta"] == 0.0 and not r["significant"]


def test_paired_clear_separation_significant() -> None:
    r = paired_bootstrap([True] * 20, [False] * 20)
    assert r["delta"] == 1.0 and r["significant"]


def test_clustered_resamples_by_paper() -> None:
    # 3 papers, A passes all claims, B fails all → delta = +1.
    a = [[True, True], [True], [True, True, True]]
    b = [[False, False], [False], [False, False, False]]
    r = clustered_bootstrap(a, b)
    assert r["n"] == 3  # n = papers, not claims
    assert r["delta"] == 1.0 and r["significant"]


def test_clustered_identical_spans_zero() -> None:
    a = [[True, False], [True], [False, True, True]]
    r = clustered_bootstrap(a, a)
    assert r["delta"] == 0.0 and not r["significant"]
