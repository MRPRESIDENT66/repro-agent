"""Reliability harness — run a manifest's hint-light task N times.

    python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5

Reports staged pass rates. Every claimed match is gated by an anti-hardcode
check: the eval script must actually load the dataset and argmax over it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import extract_number, verify
from run_repro import build_task

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def _genuine(ws: Path) -> bool:
    """Real eval = loads the dataset AND predicts over it (not a hardcoded print)."""
    for f in ws.glob("*.py"):
        src = f.read_text(errors="replace")
        has_data = "load_dataset" in src or "CIFAR" in src.upper() or "dataset" in src.lower()
        has_pred = "argmax" in src or ".max(" in src or "topk" in src
        if has_data and has_pred:
            return True
    return False


def main() -> None:
    manifest = sys.argv[1] if len(sys.argv) > 1 else "evals/benchmark/cifar10_resnet20.yaml"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    m = yaml.safe_load((ROOT / manifest).read_text())
    task = build_task(m)
    expected, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    name = Path(manifest).stem

    rows = []
    for i in range(N):
        ws = ROOT / f"workspaces/rel_{name}_{i}"
        import shutil
        shutil.rmtree(ws, ignore_errors=True)
        session = Session(ws, venv_python=REPRO_PY, default_timeout=400)
        r = run_agent(task, expected, session, ChatLLM(), max_steps=20)
        actual = extract_number(r.final_raw) if r.gave_final else None
        v = verify(actual, expected, tol)
        matched = bool(v.match and _genuine(ws))
        rows.append({"matched": matched, "metric": actual is not None,
                     "steps": r.steps, "errors": r.errors, "actual": actual})
        print(f"run {i}: matched={matched} actual={actual} steps={r.steps} errors={r.errors}")

    print(f"\n===== {name}: reliability over {N} runs (deepseek-chat) =====")
    for stage in ("metric", "matched"):
        c = sum(row[stage] for row in rows)
        print(f"  {stage:10} {c}/{N} = {c/N:.0%}")
    print(f"  avg steps={sum(r['steps'] for r in rows)/N:.1f}  "
          f"avg errors={sum(r['errors'] for r in rows)/N:.1f}")


if __name__ == "__main__":
    main()
