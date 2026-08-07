#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evals" / "runs"
RESULTS = ROOT / "evals" / "RESULTS.md"
README = ROOT / "README.md"

TASK_LABELS = {
    "distilbert_sst2": "DistilBERT SST-2",
    "detectors_resnet18_cifar100": "detectors RN18 / CIFAR-100",
    "detectors_vgg16_cifar10": "detectors VGG16-bn / CIFAR-10",
    "mmpretrain_resnet18": "mmpretrain RN18 / CIFAR-10",
    "openood_ebo": "OpenOOD EBO AUROC",
    "robustbench_carmon": "RobustBench Carmon2019",
}
PIPELINE_ORDER = ["solo", "solo-repair", "full"]
TASK_ORDER = list(TASK_LABELS)


def load_results() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RUNS.glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        data["_path"] = str(path.relative_to(ROOT))
        data["_run_dir"] = path.parent.name
        rows.append(data)
    return rows


def task_key(run_dir: str) -> str | None:
    for key in TASK_ORDER:
        if run_dir.startswith(key + "_multi_rag_") or run_dir.startswith(key + "_"):
            return key
    # Special cases from actual directory names.
    if run_dir.startswith("mmpretrain_resnet18_multi_rag_"):
        return "mmpretrain_resnet18"
    if run_dir.startswith("robustbench_carmon_"):
        return "robustbench_carmon"
    return None


def failure_reason(row: dict[str, Any]) -> str:
    verdict = row.get("verdict") or {}
    if verdict.get("match") and not row.get("workflow_error"):
        return "pass"
    if row.get("workflow_error"):
        return "workflow_error"
    reason = verdict.get("reason")
    if reason:
        return str(reason)
    diagnostics = row.get("public_contract_diagnostics") or []
    if diagnostics:
        text = " ".join(str(x) for x in diagnostics).lower()
        if "predictions" in text:
            return "contract_predictions"
        return "contract_diagnostics"
    return "unknown_failure"


def verified_without_workflow_error(row: dict[str, Any]) -> bool:
    return bool((row.get("verdict") or {}).get("match")) and not bool(
        row.get("workflow_error")
    )


def clean_among_verified(clean: int, verified: int) -> str:
    if not verified:
        return "—"
    return f"{clean}/{verified} ({100 * clean / verified:.0f}%)"


def observed_eval_executions(row: dict[str, Any]) -> float:
    """Recover counts for runs where a post-execution graph exception prevented
    LangGraph State from being returned. Every oracle execution starts with one
    audited ``python -m py_compile`` command in ``commands.sh``."""
    reported = int(row.get("eval_executions") or 0)
    command_path = ROOT / row["_path"]
    command_path = command_path.parent / "commands.sh"
    if not command_path.is_file():
        return float(reported)
    observed = sum(
        1 for line in command_path.read_text(errors="replace").splitlines()
        if "python -m py_compile " in line
    )
    return float(max(reported, observed))


def mean(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{statistics.mean(values):.2f}"


def cost(values: list[float]) -> str:
    if not values:
        return "—"
    return f"¥{statistics.mean(values):.3f}"


def fmt_failure(counter: Counter[str]) -> str:
    counter = Counter({k: v for k, v in counter.items() if k != "pass" and v})
    if not counter:
        return "—"
    return ", ".join(f"{k}×{v}" for k, v in counter.most_common())


def aggregate(rows: list[dict[str, Any]], group_prefix: str):
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run_dir = row["_run_dir"]
        if group_prefix not in run_dir:
            continue
        key = task_key(run_dir)
        if key is None:
            continue
        buckets[(key, row.get("pipeline", ""))].append(row)
    return buckets


def table_for_buckets(buckets, *, e2: bool) -> str:
    lines = []
    if e2:
        lines.append("| Task | Condition | verified | no workflow error / verified | mean cmds | mean evals | mean cost | failure modes |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        order = [
            (task, pipe)
            for task in [
                "distilbert_sst2",
                "detectors_resnet18_cifar100",
                "mmpretrain_resnet18",
                "openood_ebo",
            ]
            for pipe in PIPELINE_ORDER
        ]
    else:
        lines.append("| Task | verified | no workflow error / verified | mean cmds | mean evals | mean cost | failure modes |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        order = [(task, "full") for task in TASK_ORDER]
    for key in order:
        entries = buckets.get(key, [])
        if not entries:
            continue
        passes = sum(1 for row in entries if (row.get("verdict") or {}).get("match"))
        clean_verified = sum(
            1 for row in entries if verified_without_workflow_error(row)
        )
        clean_ratio = clean_among_verified(clean_verified, passes)
        reasons = Counter(failure_reason(row) for row in entries)
        cmds = [float(row.get("total_commands") or 0) for row in entries]
        evals = [observed_eval_executions(row) for row in entries]
        costs = [float(row.get("total_cost_yuan") or 0) for row in entries]
        if e2:
            lines.append(
                f"| {TASK_LABELS[key[0]]} | `{key[1]}` | {passes}/{len(entries)} | {clean_ratio} | {mean(cmds)} | {mean(evals)} | {cost(costs)} | {fmt_failure(reasons)} |"
            )
        else:
            lines.append(
                f"| {TASK_LABELS[key[0]]} | {passes}/{len(entries)} | {clean_ratio} | {mean(cmds)} | {mean(evals)} | {cost(costs)} | {fmt_failure(reasons)} |"
            )
    if not e2:
        entries = [row for rows in buckets.values() for row in rows]
        if entries:
            passes = sum(bool((row.get("verdict") or {}).get("match")) for row in entries)
            clean_verified = sum(
                verified_without_workflow_error(row) for row in entries
            )
            total_cost = sum(float(row.get("total_cost_yuan") or 0) for row in entries)
            lines.append(
                f"| **Total** | **{passes}/{len(entries)}** | **{clean_among_verified(clean_verified, passes)}** | — | — | **¥{total_cost:.3f} total** | — |"
            )
    return "\n".join(lines)


def condition_totals(buckets) -> str:
    lines = [
        "| Condition | verified | no workflow error / verified | mean cmds | mean evals | mean cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline in PIPELINE_ORDER:
        entries = [
            row for (task, condition), rows in buckets.items()
            if condition == pipeline for row in rows
        ]
        if not entries:
            continue
        passes = sum(bool((row.get("verdict") or {}).get("match")) for row in entries)
        clean_verified = sum(
            verified_without_workflow_error(row) for row in entries
        )
        cmds = [float(row.get("total_commands") or 0) for row in entries]
        evals = [observed_eval_executions(row) for row in entries]
        costs = [float(row.get("total_cost_yuan") or 0) for row in entries]
        lines.append(
            f"| `{pipeline}` | **{passes}/{len(entries)} ({100 * passes / len(entries):.0f}%)** | "
            f"**{clean_among_verified(clean_verified, passes)}** | "
            f"{mean(cmds)} | {mean(evals)} | {cost(costs)} |"
        )
    return "\n".join(lines)


def build_section(rows: list[dict[str, Any]]) -> str:
    e1 = aggregate(rows, "e1_n5")
    e2 = aggregate(rows, "e2_n5")
    coverage = dict(e1)
    for (task, pipeline), entries in e2.items():
        if pipeline == "full" and (task, "full") not in coverage:
            coverage[(task, "full")] = entries
    return "\n".join([
        "<!-- GENERATED_N5_START -->",
        "## E1: Six-Task Coverage N=5",
        "",
        "For the four ablation tasks this table uses the `full` rows from E2; detectors VGG16 and RobustBench are additional full-pipeline N=5 coverage cells. All 30 runs use the same frozen agent snapshot and LLM configuration.",
        "",
        table_for_buckets(coverage, e2=False) if coverage else "_No coverage runs found yet._",
        "",
        "## E2: Pipeline Ablation N=5",
        "",
        table_for_buckets(e2, e2=True) if e2 else "_No E2 N=5 runs found yet._",
        "",
        condition_totals(e2) if e2 else "",
        "<!-- GENERATED_N5_END -->",
    ])


def replace_section(path: Path, section: str) -> None:
    start = "<!-- GENERATED_N5_START -->"
    end = "<!-- GENERATED_N5_END -->"
    text = path.read_text()
    if start not in text or end not in text:
        text = text.rstrip() + "\n\n" + section + "\n"
    else:
        before, tail = text.split(start, 1)
        _, after = tail.split(end, 1)
        text = before.rstrip() + "\n\n" + section + after
    path.write_text(text)


def update_readme(section: str) -> None:
    text = README.read_text()
    summary = "\n".join([
        "- **Main N=5 results are reported in [evals/RESULTS.md](evals/RESULTS.md).** Each cell uses five repeated LLM runs and reports verified passes/runs, average commands, average eval executions, cost, and failure modes.",
    ])
    anchor = "- **Coverage:"
    if anchor in text and "Main N=5 results are reported" not in text:
        text = text.replace(anchor, summary + "\n" + anchor, 1)
    README.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = load_results()
    section = build_section(rows)
    print(section)
    if args.write:
        replace_section(RESULTS, section)
        update_readme(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
