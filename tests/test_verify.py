"""Fail-closed verifier tests."""

from __future__ import annotations

from verify.check import verify_run


def _verify(tmp_path, recompute_fn, *, expected_num_examples=100):
    return verify_run(
        tmp_path,
        expected=92.6,
        tolerance=0.1,
        metric="top1_accuracy",
        expected_num_examples=expected_num_examples,
        recompute_fn=recompute_fn,
    )


def test_recomputed_metric_matches(tmp_path) -> None:
    verdict = _verify(tmp_path, lambda _workdir: (92.6, 100))

    assert verdict.match
    assert verdict.actual == 92.6
    assert verdict.reason is None


def test_missing_or_malformed_predictions_fail_closed(tmp_path) -> None:
    missing = _verify(tmp_path, lambda _workdir: None)
    malformed = _verify(tmp_path, lambda _workdir: (float("nan"), 100))

    assert not missing.match
    assert missing.reason == "no_recomputable_predictions"
    assert not malformed.match
    assert malformed.reason == "invalid_recomputed_result"


def test_wrong_sample_count_fails_closed(tmp_path) -> None:
    verdict = _verify(tmp_path, lambda _workdir: (92.6, 99))

    assert not verdict.match
    assert verdict.reason == "count_mismatch"


def test_outside_tolerance_fails_closed(tmp_path) -> None:
    verdict = _verify(tmp_path, lambda _workdir: (90.0, 100))

    assert not verdict.match
    assert verdict.reason == "outside_tolerance"


def test_recompute_exception_fails_closed(tmp_path) -> None:
    def broken(_workdir):
        raise ValueError("bad artifact")

    verdict = _verify(tmp_path, broken)

    assert not verdict.match
    assert verdict.reason == "recompute_error:ValueError"
