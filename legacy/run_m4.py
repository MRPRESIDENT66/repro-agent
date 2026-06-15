"""M4 context-compression ablation: same hard oracle, compress off vs on.

Compression keeps system+task+recent turns full and shrinks old observations.
The honest question: does it cut the context the LLM sees on a long debug
trajectory without hurting success?
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import shutil
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"
MANIFEST = "evals/benchmark/resnet18_cifar100.yaml"
def _run(compress: bool) -> dict:
    from run_repro import build_task

    m = yaml.safe_load((ROOT / MANIFEST).read_text())
    exp, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    ws = ROOT / f"workspaces/m4_{'on' if compress else 'off'}"
    shutil.rmtree(ws, ignore_errors=True)
    session = Session(ws, venv_python=REPRO_PY, default_timeout=400)
    r = run_agent(build_task(m), session, ChatLLM(), max_steps=20, compress=compress)
    verdict = verify_run(
        session.transcript,
        session.workdir,
        expected=exp,
        tolerance=tol,
        metric=m["target"]["metric"],
        expected_num_examples=int(m["dataset"]["num_examples"]),
    )
    return {"compress": compress, "matched": verdict.match,
            "steps": r.steps, "peak_ctx_chars": r.peak_ctx_chars}


def main() -> None:
    rows = [_run(False), _run(True)]
    print("\n===== M4 context compression (resnet18_cifar100) =====")
    print(f"{'compress':9} {'matched':8} {'steps':6} {'peak_ctx_chars':14}")
    for r in rows:
        print(f"{str(r['compress']):9} {str(r['matched']):8} {r['steps']:<6} {r['peak_ctx_chars']:<14}")


if __name__ == "__main__":
    main()
