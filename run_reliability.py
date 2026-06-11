"""Reliability harness — run the same M1 task N times, report staged pass rates.

The agent is stochastic, so a single success says little. This runs N fresh
attempts and reports how often it reaches each stage. Every claimed match is
gated by an anti-hardcode check: the eval script must actually load the dataset
and argmax over it, or the "match" doesn't count.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import extract_number, verify

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"
N = 8

TASK = (
    "Reproduce the published top-1 test accuracy (in percent) of the pretrained "
    "model 'cifar10_resnet20' from the torch.hub repository "
    "'chenyaofo/pytorch-cifar-models', evaluated on the full CIFAR-10 test set "
    "(10000 images)."
)
EXPECTED, TOL = 92.60, 0.10


def _script_genuine(ws: Path) -> bool:
    """Did the agent write a real eval (not a hardcoded print of the number)?"""
    for f in ws.glob("*.py"):
        src = f.read_text(errors="replace")
        if "argmax" in src and ("load_dataset" in src or "CIFAR" in src.upper()):
            return True
    return False


def main() -> None:
    rows = []
    for i in range(N):
        ws = ROOT / f"workspaces/rel_{i}"
        if ws.exists():
            for p in ws.rglob("*"):
                p.unlink() if p.is_file() else None
        session = Session(ws, venv_python=REPRO_PY, default_timeout=300)
        r = run_agent(TASK, EXPECTED, session, ChatLLM(), max_steps=15)

        actual = extract_number(r.final_raw) if r.gave_final else None
        out = "\n".join(x.stdout for x in session.transcript)
        printed_acc = any(10.0 <= float(n) <= 100.0 for n in re.findall(r"\d+\.\d+", out))
        v = verify(actual, EXPECTED, TOL)
        genuine = _script_genuine(ws)
        matched = bool(v.match and genuine)

        rows.append({
            "started": len(session.transcript) > 0,
            "ran_eval": printed_acc,
            "metric": actual is not None,
            "matched": matched,
            "steps": r.steps, "errors": r.errors, "actual": actual, "genuine": genuine,
        })
        print(f"run {i}: matched={matched} actual={actual} steps={r.steps} "
              f"errors={r.errors} genuine={genuine}")

    print(f"\n===== reliability over {N} runs (deepseek-chat) =====")
    for stage in ("started", "ran_eval", "metric", "matched"):
        c = sum(row[stage] for row in rows)
        print(f"  {stage:10} {c}/{N} = {c/N:.0%}")
    cheats = sum(row["matched"] is False and row["metric"] and row["actual"]
                 and abs(row["actual"] - EXPECTED) <= TOL and not row["genuine"] for row in rows)
    if cheats:
        print(f"  (!) {cheats} run(s) printed the right number WITHOUT a genuine eval — excluded")
    print(f"  avg steps={sum(r['steps'] for r in rows)/N:.1f}  "
          f"avg errors={sum(r['errors'] for r in rows)/N:.1f}")


if __name__ == "__main__":
    main()
