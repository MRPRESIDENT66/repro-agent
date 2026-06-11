"""Bootstrap CIs for reproduction-rate comparisons.

Two variants:

* :func:`paired_bootstrap` — per-task paired bootstrap (carried over from the
  insight-agent project).
* :func:`clustered_bootstrap` — **resamples by paper, not by claim**. Multiple
  claims from one paper share the repo/environment and are correlated; treating
  them as independent inflates n and underestimates variance. With only ~10
  papers the CI will be honestly wide — that's the signal, not a bug.

With this few papers, lean on effect sizes + staged pass-rates + the failure
taxonomy; significance is secondary and a CI spanning 0 is reported as-is.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: list[bool], b: list[bool], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Per-task paired bootstrap CI on the success-rate delta mean(a) - mean(b)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must cover the same tasks (equal length)")
    d = a_arr - b_arr
    n = len(d)
    if n == 0:
        raise ValueError("no tasks to compare")
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return _summary(n, float(d.mean()), boot, lo, hi)


def clustered_bootstrap(
    a_by_paper: list[list[bool]],
    b_by_paper: list[list[bool]],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paper-clustered paired bootstrap: resample whole papers with replacement.

    ``a_by_paper[i]`` / ``b_by_paper[i]`` are the per-claim pass indicators for
    paper ``i`` under configs A / B. The statistic is the overall success-rate
    delta; resampling is at the paper level to respect within-paper correlation.
    """
    if len(a_by_paper) != len(b_by_paper):
        raise ValueError("a and b must cover the same papers")
    papers = len(a_by_paper)
    if papers == 0:
        raise ValueError("no papers to compare")

    def rate_delta(idx: np.ndarray) -> float:
        a_hits = a_tot = b_hits = b_tot = 0
        for i in idx:
            a_hits += sum(a_by_paper[i]); a_tot += len(a_by_paper[i])
            b_hits += sum(b_by_paper[i]); b_tot += len(b_by_paper[i])
        return (a_hits / a_tot if a_tot else 0.0) - (b_hits / b_tot if b_tot else 0.0)

    observed = rate_delta(np.arange(papers))
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rate_delta(rng.integers(0, papers, size=papers)) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return _summary(papers, observed, boot, lo, hi)


def _summary(n: int, delta: float, boot: np.ndarray, lo: float, hi: float) -> dict:
    return {
        "n": n,
        "delta": delta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float((boot > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
    }
