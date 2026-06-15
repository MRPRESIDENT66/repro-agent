"""Generate the result tables straight from `evals/runs/*/result.json`.

Single source of truth for the README / FINAL_REPORT, so the prose can never
drift from the actual artifacts. Run from the repo root:

    python evals/report_tables.py            # print E1 + E2 markdown tables
    python evals/report_tables.py --check     # exit non-zero if any run is missing

Nothing here recomputes a metric — it only counts what the blind verifier already
wrote (``verdict.match``) per run.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "evals" / "runs"

# (label, run-dir slug, published target, blind level). Blind level is a property
# of the task's provisioning, not of any single run:
#   strict = target absent from the workspace; soft = present in the public repo
#   (e.g. mmpretrain's model-zoo metafile) but never shown by task/verifier.
E1_TASKS = [
    ("DistilBERT SST-2", "distilbert_sst2_multi_rag", "91.06 acc", "strict"),
    ("mmpretrain ResNet-18", "mmpretrain_resnet18_multi_rag", "94.82 top-1", "soft"),
    ("detectors ResNet-18 C100", "detectors_resnet18_cifar100_multi_rag", "79.26 top-1", "strict"),
    ("OpenOOD EBO", "openood_ebo_multi_rag", "87.58 AUROC", "strict"),
    ("RobustBench Carmon2019", "robustbench_carmon", "52.0 robust acc", "strict"),
]
E2_TASKS = [("easy — DistilBERT", "distilbert_sst2_multi_rag"),
            ("hard — OpenOOD", "openood_ebo_multi_rag")]
E2_CONDITIONS = ["solo", "team", "solo-retry", "solo-repair", "full"]
REPS = ["n1", "n2", "n3", "n4", "n5"]


def _load(slug: str, pipeline: str) -> list[dict]:
    suffix = "" if pipeline == "full" else f"__{pipeline}"
    out = []
    for rep in REPS:
        p = RUNS / f"{slug}_{rep}{suffix}" / "result.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def _n_exec(r: dict) -> int:
    # Prefer the recorded budget consumption; otherwise derive it from the
    # follow-up roles (1 initial execution + one per repair/retry round) so runs
    # predating the eval_executions field still report accurately.
    if "eval_executions" in r:
        return r["eval_executions"]
    return 1 + sum(k.startswith(("repair_", "retry_")) for k in r["roles"])


def _agg(runs: list[dict]) -> dict:
    n = len(runs)
    if not n:
        return {"n": 0}
    passed = sum(r["verdict"]["match"] for r in runs)
    repair = sum(any(k.startswith(("repair_", "retry_")) for k in r["roles"]) for r in runs)
    cost = sum(r.get("total_cost_yuan", 0.0) for r in runs) / n
    execs = sum(_n_exec(r) for r in runs) / n
    return {"n": n, "pass": passed, "repair": repair, "cost": cost, "execs": execs}


def e1_table() -> str:
    rows = ["| Task | blind | passed (full, N=5) | repair fired | ~cost/run |",
            "|---|---|---|---|---|"]
    tot_p = tot_n = 0
    for label, slug, target, blind in E1_TASKS:
        a = _agg(_load(slug, "full"))
        if not a["n"]:
            rows.append(f"| {label} ({target}) | {blind} | — pending — | | |")
            continue
        tot_p += a["pass"]; tot_n += a["n"]
        rows.append(
            f"| {label} ({target}) | {blind} | **{a['pass']}/{a['n']}** | "
            f"{a['repair']}/{a['n']} | ¥{a['cost']:.3f} |"
        )
    rows.append(f"| **total** | | **{tot_p}/{tot_n}** | | |")
    return "\n".join(rows)


def e2_table() -> str:
    rows = ["| condition | " + " | ".join(label for label, _ in E2_TASKS) + " | exec/run | cost/run |",
            "|---|" + "---|" * (len(E2_TASKS) + 2)]
    for cond in E2_CONDITIONS:
        cells = []
        execs = costs = 0.0
        seen = 0
        for _, slug in E2_TASKS:
            a = _agg(_load(slug, cond))
            cells.append(f"{a['pass']}/{a['n']}" if a["n"] else "—")
            if a["n"]:
                execs += a["execs"]; costs += a["cost"]; seen += 1
        ex = f"{execs/seen:.1f}" if seen else "—"
        co = f"¥{costs/seen:.3f}" if seen else "—"
        rows.append(f"| {cond} | " + " | ".join(cells) + f" | {ex} | {co} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any run is missing")
    args = ap.parse_args()

    print("## E1 — feasibility & generality (full pipeline, N=5, blind)\n")
    print(e1_table())
    print("\n## E2 — equal-budget ablation (≤5 executions per condition, N=5)\n")
    print(e2_table())

    if args.check:
        missing = []
        for _, slug, _, _ in E1_TASKS:
            if len(_load(slug, "full")) < len(REPS):
                missing.append(f"{slug} full")
        for _, slug in E2_TASKS:
            for cond in E2_CONDITIONS:
                if len(_load(slug, cond)) < len(REPS):
                    missing.append(f"{slug} {cond}")
        if missing:
            print("\nMISSING (pending):")
            for m in missing:
                print(f"  - {m}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
