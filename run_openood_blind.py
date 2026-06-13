"""Run one strict-blind OpenOOD EBO Agent experiment.

The Agent sees only a clean fixed OpenOOD tree plus official data/checkpoints in
an offline CPU Docker container. The canonical runner, canonical result,
experiment note, manifest, and private verifier remain outside the mount.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.docker_session import DockerSession
from verify.check import verify_run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "repos" / "OpenOOD"
WORKDIR = ROOT / "workspaces" / "openood_ebo_blind"
IMAGE = "repro-openood:latest"
ATTEMPT = os.environ.get("OPENOOD_ATTEMPT", "001")
MAX_STEPS = int(os.environ.get("OPENOOD_MAX_STEPS", "30"))
ARTIFACT_DIR = ROOT / "evals" / "runs" / f"openood_ebo_blind_{ATTEMPT}"

EXPECTED = 87.58
TOLERANCE = 0.05
EXPECTED_DATASETS = {"cifar100": 9000, "tin": 7793}
EXPECTED_RUNS = ["s0", "s1", "s2"]
AGGREGATION = "dataset_mean_then_run_mean"

TASK = """You are in the root of a clean fixed OpenOOD repository. Official
CIFAR-10 CrossEntropy ResNet-18 checkpoints for runs s0, s1, and s2 are already
under results/, and the official CIFAR-10 benchmark data needed for Near-OOD
evaluation is already under data/.

Reproduce the official Energy-based OOD detection (EBO) Near-OOD AUROC for
CIFAR-10 using all three official runs and both Near-OOD datasets, CIFAR-100 and
TinyImageNet. Use the repository's definitions and benchmark image lists. The
environment is CPU-only and fully offline; diagnose and adapt repository code
when necessary without changing the evaluation semantics. Report percentage
AUROC, not a 0-1 fraction."""

EVIDENCE = """A result counts only when an EXECUTED evaluation command prints
one machine-readable line in exactly this shape:
  REPRO_RESULT {"metric":"near_ood_auroc","actual":<number>,
  "datasets":{"cifar100":<count>,"tin":<count>},
  "run_metrics":{"s0":{"cifar100":<auroc>,"tin":<auroc>},
  "s1":{"cifar100":<auroc>,"tin":<auroc>},
  "s2":{"cifar100":<auroc>,"tin":<auroc>}},
  "aggregation":"dataset_mean_then_run_mean"}
The evaluation program itself must print the line. Do not echo/printf it
afterward. `actual` must equal the mean of the two dataset AUROCs within each
run, then the mean across all three runs. Keep full precision in the evidence."""


def _copy_clean_source() -> None:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    shutil.copytree(
        SOURCE,
        WORKDIR,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "run_nearood_ebo_cpu.py",
            "nearood_ebo_cpu_results.json",
        ),
    )


def _assert_blind_workspace() -> None:
    forbidden_names = {
        "run_nearood_ebo_cpu.py",
        "nearood_ebo_cpu_results.json",
        "OPENOOD_EBO.md",
    }
    present = {p.name for p in WORKDIR.rglob("*") if p.is_file()}
    leaked_names = forbidden_names & present
    if leaked_names:
        raise RuntimeError(f"private files leaked into blind workspace: {leaked_names}")

    for path in WORKDIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".csv", ".sh"
        }:
            continue
        text = path.read_text(errors="replace")
        if "87.58" in text:
            raise RuntimeError(f"private target leaked into blind workspace: {path}")


def main() -> None:
    _copy_clean_source()
    _assert_blind_workspace()

    session = DockerSession(WORKDIR, image=IMAGE, mem="6g", cpus=6.0,
                            default_timeout=1800)
    session.go_offline()
    try:
        result = run_agent(
            TASK,
            session,
            ChatLLM(),
            max_steps=MAX_STEPS,
            compress=True,
            evidence_instructions=EVIDENCE,
        )
    finally:
        session.close()

    verdict = verify_run(
        session.transcript,
        session.workdir,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        metric="near_ood_auroc",
        expected_num_examples=None,
        expected_datasets=EXPECTED_DATASETS,
        expected_runs=EXPECTED_RUNS,
        expected_aggregation=AGGREGATION,
    )
    output = {
        "task": TASK,
        "attempt": ATTEMPT,
        "max_steps": MAX_STEPS,
        "blind_workspace_checked": True,
        "steps": result.steps,
        "errors": result.errors,
        "format_errors": result.format_errors,
        "gave_final": result.gave_final,
        "usage": result.usage,
        "peak_ctx_tokens": result.peak_ctx_tokens,
        "verdict": verdict.as_dict(),
        "commands": [r.command for r in session.transcript],
    }
    result_json = json.dumps(output, indent=2) + "\n"
    replay_script = session.replay_script() + "\n"
    transcript_jsonl = "".join(
        json.dumps(message) + "\n" for message in result.transcript
    )
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "blind_result.json").write_text(result_json)
        (output_dir / "commands.sh").write_text(replay_script)
        (output_dir / "transcript.jsonl").write_text(transcript_jsonl)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
