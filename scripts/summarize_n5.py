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
    if row.get("workflow_error"):
        return "workflow_error"
    verdict = row.get("verdict") or {}
    if verdict.get("match"):
        return "pass"
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
        lines.append("| Task | Condition | pass@5 | mean cmds | mean evals | mean cost | failure modes |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        order = [(task, pipe) for task in ["distilbert_sst2", "detectors_resnet18_cifar100"] for pipe in PIPELINE_ORDER]
    else:
        lines.append("| Task | pass@5 | mean cmds | mean evals | mean cost | failure modes |")
        lines.append("|---|---:|---:|---:|---:|---|")
        order = [(task, "full") for task in TASK_ORDER]
    for key in order:
        entries = buckets.get(key, [])
        if not entries:
            continue
        passes = sum(1 for row in entries if (row.get("verdict") or {}).get("match"))
        reasons = Counter(failure_reason(row) for row in entries)
        cmds = [float(row.get("total_commands") or 0) for row in entries]
        evals = [float(row.get("eval_executions") or 0) for row in entries]
        costs = [float(row.get("total_cost_yuan") or 0) for row in entries]
        if e2:
            lines.append(
                f"| {TASK_LABELS[key[0]]} | `{key[1]}` | {passes}/{len(entries)} | {mean(cmds)} | {mean(evals)} | {cost(costs)} | {fmt_failure(reasons)} |"
            )
        else:
            lines.append(
                f"| {TASK_LABELS[key[0]]} | {passes}/{len(entries)} | {mean(cmds)} | {mean(evals)} | {cost(costs)} | {fmt_failure(reasons)} |"
            )
    return "\n".join(lines)


def build_section(rows: list[dict[str, Any]]) -> str:
    e1 = aggregate(rows, "e1_n5")
    e2 = aggregate(rows, "e2_n5")
    return "\n".join([
        "## Main N=5 Results",
        "",
        "Each cell is five independent LLM runs under the same model, prompts, execution budget, and verifier. We report pass@5, mean command count, mean evaluation executions, mean LLM cost, and verifier-level failure modes.",
        "",
        "### E1 — Coverage N=5",
        "",
        table_for_buckets(e1, e2=False) if e1 else "_No E1 N=5 runs found yet._",
        "",
        "### E2 — Pipeline Ablation N=5",
        "",
        table_for_buckets(e2, e2=True) if e2 else "_No E2 N=5 runs found yet._",
        "",
    ])


def replace_section(path: Path, section: str) -> None:
    marker = "## Main N=5 Results"
    text = path.read_text()
    if marker not in text:
        text = text.rstrip() + "\n\n" + section + "\n"
    else:
        before = text.split(marker, 1)[0].rstrip()
        text = before + "\n\n" + section + "\n"
    path.write_text(text)


def update_readme(section: str) -> None:
    text = README.read_text()
    summary = "\n".join([
        "- **Main N=5 results are reported in [evals/RESULTS.md](evals/RESULTS.md).** Each cell uses five independent LLM runs and reports pass@5, average commands, average eval executions, cost, and failure modes.",
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
