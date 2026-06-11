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
    metric = m["target"]["metric"].replace("top1_accuracy", "top-1 accuracy").replace("_", " ")
    return (
        f"Reproduce the published {metric} (in percent) of {model_desc} "
        f"on the {m['dataset']['name']}."
    )


def reproduce(manifest_path: str) -> dict:
    """Run the full agent → verify pipeline on one manifest; return structured result."""
    m = yaml.safe_load((ROOT / manifest_path).read_text())
    expected, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    name = Path(manifest_path).stem
    task = build_task(m)

    session = Session(ROOT / f"workspaces/{name}", venv_python=REPRO_PY, default_timeout=400)
    result = run_agent(task, expected, session, ChatLLM(), max_steps=20)

    actual = extract_number(result.final_raw) if result.gave_final else None
    all_stdout = "\n".join(r.stdout for r in session.transcript)
    evidence = find_evidence(all_stdout, actual) if actual is not None else None
    v = verify(actual, expected, tol, evidence)
    printed_acc = any(10.0 <= float(n) <= 100.0 for n in re.findall(r"\d+\.\d+", all_stdout))
    stages = {
        "repo_inspected": len(session.transcript) > 0,
        "evaluation_completed": printed_acc,
        "metric_extracted": actual is not None,
        "claim_matched": v.match,
    }
    scripts = sorted(session.workdir.glob("*.py"), key=lambda p: p.stat().st_mtime)
    eval_script = scripts[-1].read_text(errors="replace") if scripts else ""

    return {
        "name": name, "task": task, "stages": stages, "verdict": v.as_dict(),
        "steps": result.steps, "errors": result.errors,
        "commands": [r.command.splitlines()[0][:100] for r in session.transcript],
        "eval_script": eval_script,
    }


def main() -> None:
    manifest = sys.argv[1] if len(sys.argv) > 1 else "evals/benchmark/cifar10_resnet20.yaml"
    r = reproduce(manifest)
    print(f"\n========== {r['name']} ==========")
    print(f"task: {r['task']}")
    print(f"steps={r['steps']} errors={r['errors']}")
    for k, ok in r["stages"].items():
        print(f"  [{'x' if ok else ' '}] {k}")
    print(f"verdict: {r['verdict']}")
    for i, c in enumerate(r["commands"], 1):
        print(f"  {i}. {c}")


if __name__ == "__main__":
    main()
