"""M4 context-compression ablation: same hard oracle, compress off vs on.

Compression keeps system+task+recent turns full and shrinks old observations.
The honest question: does it cut the context the LLM sees on a long debug
trajectory without hurting success?
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import extract_number, verify

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"
MANIFEST = "evals/benchmark/resnet18_cifar100.yaml"
_ACC = re.compile(r"(\d{2}\.\d+)")


def _run(compress: bool) -> dict:
    from run_repro import build_task

    m = yaml.safe_load((ROOT / MANIFEST).read_text())
    exp, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    ws = ROOT / f"workspaces/m4_{'on' if compress else 'off'}"
    shutil.rmtree(ws, ignore_errors=True)
    session = Session(ws, venv_python=REPRO_PY, default_timeout=400)
    r = run_agent(build_task(m), exp, session, ChatLLM(), max_steps=20, compress=compress)
    out = "\n".join(x.stdout for x in session.transcript)
    actual = extract_number(r.final_raw) if r.gave_final else (
        next((float(n) for n in _ACC.findall(out) if abs(float(n) - exp) <= tol), None))
    return {"compress": compress, "matched": verify(actual, exp, tol).match,
            "steps": r.steps, "peak_ctx_chars": r.peak_ctx_chars}


def main() -> None:
    rows = [_run(False), _run(True)]
    print("\n===== M4 context compression (resnet18_cifar100) =====")
    print(f"{'compress':9} {'matched':8} {'steps':6} {'peak_ctx_chars':14}")
    for r in rows:
        print(f"{str(r['compress']):9} {str(r['matched']):8} {r['steps']:<6} {r['peak_ctx_chars']:<14}")


if __name__ == "__main__":
    main()
