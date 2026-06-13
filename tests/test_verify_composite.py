"""Composite evidence for multi-dataset, multi-run reproduction tasks."""

from __future__ import annotations

from exec.session import RunResult
from verify.check import verify_transcript


def _run(line: str) -> RunResult:
    return RunResult("python eval.py", line, "", 0, False, 1.0)


def _verify(line: str):
    return verify_transcript(
        [_run(line)],
        expected=87.58,
        tolerance=0.05,
        metric="near_ood_auroc",
        expected_num_examples=None,
        expected_datasets={"cifar100": 9000, "tin": 7793},
        expected_runs=["s0", "s1", "s2"],
        expected_aggregation="dataset_mean_then_run_mean",
    )


def test_composite_evidence_recomputes_aggregate() -> None:
    line = (
        'REPRO_RESULT {"metric":"near_ood_auroc","actual":87.58,'
        '"datasets":{"cifar100":9000,"tin":7793},'
        '"run_metrics":{"s0":{"cifar100":85.54,"tin":88.32},'
        '"s1":{"cifar100":86.88,"tin":88.94},'
        '"s2":{"cifar100":86.66,"tin":89.14}},'
        '"aggregation":"dataset_mean_then_run_mean"}'
    )
    verdict = _verify(line)
    assert verdict.match
    assert verdict.datasets == {"cifar100": 9000, "tin": 7793}


def test_composite_rejects_claim_not_supported_by_components() -> None:
    line = (
        'REPRO_RESULT {"metric":"near_ood_auroc","actual":87.58,'
        '"datasets":{"cifar100":9000,"tin":7793},'
        '"run_metrics":{"s0":{"cifar100":1,"tin":1},'
        '"s1":{"cifar100":1,"tin":1},"s2":{"cifar100":1,"tin":1}},'
        '"aggregation":"dataset_mean_then_run_mean"}'
    )
    assert not _verify(line).match


def test_composite_accepts_two_decimal_component_rounding() -> None:
    line = (
        'REPRO_RESULT {"metric":"near_ood_auroc","actual":87.58,'
        '"datasets":{"cifar100":9000,"tin":7793},'
        '"run_metrics":{"s0":{"cifar100":85.55,"tin":88.33},'
        '"s1":{"cifar100":86.89,"tin":88.95},'
        '"s2":{"cifar100":86.67,"tin":89.15}},'
        '"aggregation":"dataset_mean_then_run_mean"}'
    )
    assert _verify(line).match


def test_composite_rejects_missing_run_or_wrong_dataset_count() -> None:
    line = (
        'REPRO_RESULT {"metric":"near_ood_auroc","actual":87.58,'
        '"datasets":{"cifar100":9000,"tin":100},'
        '"run_metrics":{"s0":{"cifar100":86,"tin":89},'
        '"s1":{"cifar100":86,"tin":89}},'
        '"aggregation":"dataset_mean_then_run_mean"}'
    )
    assert not _verify(line).match
