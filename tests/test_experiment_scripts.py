from scripts.run_n5 import build_specs
from scripts.summarize_n5 import condition_totals, table_for_buckets


def test_e2_runner_covers_harder_tasks_and_three_conditions() -> None:
    specs = [spec for spec in build_specs(2, include_robustbench=False) if spec.group == "e2_n5"]

    assert len(specs) == 2 * 4 * 3
    assert {spec.task for spec in specs} == {
        "distilbert",
        "detectors_rn18",
        "mmpretrain",
        "openood",
    }
    assert {spec.pipeline for spec in specs} == {"solo", "solo-repair", "full"}


def test_summary_uses_passes_over_runs_not_pass_at_k() -> None:
    table = table_for_buckets({}, e2=True)

    assert "verified" in table
    assert "no workflow error / verified" in table
    assert "pass@" not in table


def test_summary_reports_workflow_reliability_among_verified_runs() -> None:
    rows = [
        {"_path": "missing/1.json", "verdict": {"match": True}, "workflow_error": None},
        {"_path": "missing/2.json", "verdict": {"match": True}, "workflow_error": "review failed"},
        {"_path": "missing/3.json", "verdict": {"match": False}, "workflow_error": None},
    ]

    table = condition_totals({("distilbert_sst2", "solo"): rows})

    assert "**2/3 (67%)**" in table
    assert "**1/2 (50%)**" in table
