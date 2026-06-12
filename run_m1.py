"""M1 runner — let the agent autonomously reproduce one oracle, end to end.

Orchestrator (this process) needs the OpenAI-compatible client; the agent's shell uses a
SEPARATE per-task env (the oracle venv with torch/datasets) — exactly the
design's orchestrator-env ≠ repro-env split.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from agent.llm import DashScopeLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"

TASK = (
    "Reproduce the published top-1 test accuracy (in percent) of the pretrained "
    "model 'cifar10_resnet20' from the torch.hub repository "
    "'chenyaofo/pytorch-cifar-models', evaluated on the full CIFAR-10 test set "
    "(10000 images). Use the machine-readable metric id 'top1_accuracy' in "
    "REPRO_RESULT."
)


def main() -> None:
    m = yaml.safe_load((ROOT / "evals/benchmark/cifar10_resnet20.yaml").read_text())
    expected = float(m["target"]["expected"])
    tol = float(m["target"]["tolerance"])

    workdir = ROOT / "workspaces/m1_run"
    shutil.rmtree(workdir, ignore_errors=True)
    session = Session(workdir, venv_python=REPRO_PY, default_timeout=300)
    result = run_agent(TASK, session, DashScopeLLM(), max_steps=15)

    v = verify_run(
        session.transcript,
        session.workdir,
        expected=expected,
        tolerance=tol,
        metric=m["target"]["metric"],
        expected_num_examples=int(m["dataset"]["num_examples"]),
    )
    # 7-stage progress (M1 subset: the env is pre-provisioned)
    stages = {
        "evaluation_started": len(session.transcript) > 0,
        "evaluation_completed": v.evidence_line is not None,
        "metric_extracted": v.actual is not None,
        "claim_matched": v.match,
    }

    # Full transcript to disk — what the agent said, ran, and saw (observability).
    log = session.workdir / "transcript.txt"
    with log.open("w") as f:
        for msg in result.transcript:
            f.write(f"\n{'='*70}\n[{msg['role'].upper()}]\n{msg['content']}\n")
    print(f"\n(full transcript → {log})")

    print("\n========== M1 RESULT ==========")
    print(f"steps={result.steps}  errors={result.errors}  gave_final={result.gave_final}")
    for k, ok in stages.items():
        print(f"  [{'x' if ok else ' '}] {k}")
    print(f"\nverdict: {v.as_dict()}")
    print(f"\nreplayable command sequence ({len(session.transcript)} cmds):")
    for i, r in enumerate(session.transcript, 1):
        print(f"  {i}. [{ 'OK' if r.ok else 'ERR'}] {r.command.splitlines()[0][:80]}")


if __name__ == "__main__":
    main()
