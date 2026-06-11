"""Multi-agent vs single-agent ablation on a multi-result oracle.

    python run_multiagent.py

Honest question (per the design): multi-agent's value is context/role ISOLATION,
not success or speed. This runs both and reports success, the max context any one
agent reached (the isolation property), and total LLM steps (the cost).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agents.multi import run_multi, run_single

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def _genuine(root: Path) -> bool:
    for f in root.rglob("*.py"):
        s = f.read_text(errors="replace")
        if ("argmax" in s or ".max(" in s) and ("load_dataset" in s or "CIFAR" in s.upper()):
            return True
    return False


def main() -> None:
    m = yaml.safe_load((ROOT / "evals/benchmark/cifar10_resnet_multi.yaml").read_text())
    n = len(m["subtargets"])
    ws = ROOT / "workspaces/multiagent"
    import shutil
    shutil.rmtree(ws, ignore_errors=True)
    llm = ChatLLM()

    multi = run_multi(m, ws, REPRO_PY, llm)
    single = run_single(m, ws, REPRO_PY, llm)

    print(f"\n===== multi-agent vs single ({n} CIFAR-10 ResNets) =====")
    print(f"{'mode':8} {'agents':6} {'matched':8} {'max_ctx_msgs':12} {'max_ctx_chars':14} {'steps':6} {'genuine':7}")
    genuine = {
        "multi": any(_genuine(ws / f"multi_{s['model']}") for s in m["subtargets"]),
        "single": _genuine(ws / "single"),
    }
    for r in (multi, single):
        print(f"{r['mode']:8} {r['n_agents']:<6} {r['matched']}/{n:<6} "
              f"{r['max_ctx_msgs']:<12} {r['max_ctx_chars']:<14} {r['total_steps']:<6} {genuine[r['mode']]}")

    print("\nper-sub (multi):")
    for x in multi["results"]:
        print(f"  {x.model:18} actual={x.actual} matched={x.matched} steps={x.steps}")
    print("isolation = max context any single agent reached (smaller = better isolated).")


if __name__ == "__main__":
    main()
