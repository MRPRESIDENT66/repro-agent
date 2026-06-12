"""Run the agent on a benchmark manifest. Hint-light: the task is built from the
manifest's model/dataset/claim only — never the loading mechanism or gotchas.

    python run_repro.py evals/benchmark/resnet18_cifar100.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def build_task(m: dict) -> str:
    """A fair, hint-light task: what to reproduce, not how."""
    if m.get("hf_model"):
        model_desc = f"the HuggingFace model '{m['hf_model']}'"
    else:
        model_desc = f"the model '{m['model']}' from the torch.hub repository '{m['repo']}'"
    metric_id = m["target"]["metric"]
    metric = metric_id.replace("top1_accuracy", "top-1 accuracy").replace("_", " ")
    return (
        f"Reproduce the published {metric} (in percent) of {model_desc} "
        f"on the {m['dataset']['name']}. Use the machine-readable metric id "
        f"'{metric_id}' in REPRO_RESULT."
    )


def reproduce(manifest_path: str) -> dict:
    """Run the full agent → verify pipeline on one manifest; return structured result."""
    m = yaml.safe_load((ROOT / manifest_path).read_text())
    expected, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    name = Path(manifest_path).stem
    task = build_task(m)

    workdir = ROOT / f"workspaces/{name}"
    # A prior run's audit files contain the private verdict. Start clean so the
    # next Agent cannot discover the expected value from stale workspace state.
    shutil.rmtree(workdir, ignore_errors=True)
    session = Session(workdir, venv_python=REPRO_PY, default_timeout=400)
    result = run_agent(task, session, ChatLLM(), max_steps=20)

    v = verify_run(
        session.transcript,
        session.workdir,
        expected=expected,
        tolerance=tol,
        metric=m["target"]["metric"],
        expected_num_examples=int(m["dataset"]["num_examples"]),
    )
    stages = {
        "repo_inspected": len(session.transcript) > 0,
        "evaluation_completed": v.evidence_line is not None,
        "metric_extracted": v.actual is not None,
        "claim_matched": v.match,
    }
    scripts = sorted(session.workdir.glob("*.py"), key=lambda p: p.stat().st_mtime)
    eval_script = scripts[-1].read_text(errors="replace") if scripts else ""

    output = {
        "name": name, "task": task, "stages": stages, "verdict": v.as_dict(),
        "steps": result.steps, "errors": result.errors,
        "commands": [r.command.splitlines()[0][:100] for r in session.transcript],
        "eval_script": eval_script,
    }
    (session.workdir / "result.json").write_text(json.dumps(output, indent=2))
    (session.workdir / "commands.sh").write_text(session.replay_script() + "\n")
    with (session.workdir / "transcript.jsonl").open("w") as f:
        for message in result.transcript:
            f.write(json.dumps(message) + "\n")
    return output


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
