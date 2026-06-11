"""Run the agent on a benchmark manifest. Hint-light: the task is built from the
manifest's model/dataset/claim only — never the loading mechanism or gotchas.

    python run_repro.py evals/benchmark/resnet18_cifar100.yaml
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import extract_number, find_evidence, verify

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def build_task(m: dict) -> str:
    """A fair, hint-light task: what to reproduce, not how."""
    if m.get("hf_model"):
        model_desc = f"the HuggingFace model '{m['hf_model']}'"
    else:
        model_desc = f"the model '{m['model']}' from the torch.hub repository '{m['repo']}'"
    return (
        f"Reproduce the published top-1 test accuracy (in percent) of {model_desc} "
        f"on the {m['dataset']['name']}."
    )


def main() -> None:
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "evals/benchmark/cifar10_resnet20.yaml"
    m = yaml.safe_load((ROOT / manifest_path).read_text())
    expected = float(m["target"]["expected"])
    tol = float(m["target"]["tolerance"])
    name = Path(manifest_path).stem

    session = Session(ROOT / f"workspaces/{name}", venv_python=REPRO_PY, default_timeout=400)
    result = run_agent(build_task(m), expected, session, ChatLLM(), max_steps=15)

    actual = extract_number(result.final_raw) if result.gave_final else None
    all_stdout = "\n".join(r.stdout for r in session.transcript)
    evidence = find_evidence(all_stdout, actual) if actual is not None else None
    v = verify(actual, expected, tol, evidence)
    printed_acc = any(10.0 <= float(n) <= 100.0 for n in re.findall(r"\d+\.\d+", all_stdout))
    stages = {
        "evaluation_started": len(session.transcript) > 0,
        "evaluation_completed": printed_acc,
        "metric_extracted": actual is not None,
        "claim_matched": v.match,
    }

    (session.workdir / "transcript.txt").write_text(
        "".join(f"\n{'='*70}\n[{x['role'].upper()}]\n{x['content']}\n" for x in result.transcript)
    )
    print(f"\n========== {name} ==========")
    print(f"task: {build_task(m)}")
    print(f"steps={result.steps} errors={result.errors} gave_final={result.gave_final}")
    for k, ok in stages.items():
        print(f"  [{'x' if ok else ' '}] {k}")
    print(f"verdict: {v.as_dict()}")
    for i, r in enumerate(session.transcript, 1):
        print(f"  {i}. [{'OK' if r.ok else 'ERR'}] {r.command.splitlines()[0][:85]}")


if __name__ == "__main__":
    main()
