"""Multi-agent ablations on a multi-result oracle.

    python run_multiagent.py

Two honest questions (per the design):
  1. ISOLATION — multi-agent's value is context/role isolation, not success.
     Reported: success, the max context any one agent reached, steps, cost.
  2. CONCURRENCY — the sub-tasks are independent, so does running them in
     parallel cut wall-clock, or does single-box contention (MPS/RAM) eat it?
     Reported: multi-parallel vs multi-serial vs single wall-clock — as-is.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import shutil
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
    shutil.rmtree(ws, ignore_errors=True)

    rows = [
        run_multi(m, ws / "par", REPRO_PY, ChatLLM, parallel=True),
        run_multi(m, ws / "ser", REPRO_PY, ChatLLM, parallel=False),
        run_single(m, ws / "one", REPRO_PY, ChatLLM),
    ]

    print(f"\n===== multi-agent ablations ({n} CIFAR-10 ResNets) =====")
    hdr = f"{'mode':15} {'agents':6} {'matched':8} {'max_ctx_chars':14} {'steps':6} {'cost¥':8} {'wall_s':7}"
    print(hdr)
    for r in rows:
        print(f"{r['mode']:15} {r['n_agents']:<6} {r['matched']}/{n:<6} "
              f"{r['max_ctx_chars']:<14} {r['total_steps']:<6} "
              f"{r['total_cost_yuan']:<8} {r['wall_s']:<7}")

    print("\nisolation = max context any single agent reached (smaller = better isolated).")
    print("concurrency = wall_s of multi-parallel vs multi-serial (watch for MPS/RAM contention).")


if __name__ == "__main__":
    main()
