"""mmpretrain ResNet-18 CIFAR-10 oracle configuration for the multi-RAG orchestration.

Image-classification domain in the OpenMMLab / mmcv ecosystem. Unlike the
library-load oracles, the agent is dropped into a cloned ~1800-file research repo
and must NAVIGATE to the eval entry (tools/test.py) and the right config, then
write a small wrapper that runs it and parses mmengine's printed top-1 accuracy.
Runs in the pre-provisioned linux/amd64 Docker image (repro-mmpretrain:latest)
with torch2.1.0-cpu + a prebuilt mmcv wheel; checkpoint and CIFAR-10 are
provisioned on disk and the container is taken offline before execution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.types import OracleConfig
from exec.docker_session import DockerSession

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "mmpretrain"
# CIFAR-10 test gold labels (standard test order, which mmpretrain's test loader
# also follows — shuffle is off for test).
GOLD_LABELS = ROOT / "evals" / "oracles" / "gold" / "cifar10_test_labels.json"
# Provisioned assets live in a stable (gitignored) dir, like robustbench_models/
# — not under the transient workspaces/ tree that gets wiped between runs.
CKPT_SOURCE = ROOT / "repos" / "mmpretrain_assets" / "ckpt.pth"
CIFAR_SOURCE = ROOT / "repos" / "mmpretrain_assets" / "cifar10"
IMAGE = "repro-mmpretrain:latest"

EXPECTED = 94.82
TOLERANCE = 0.10
METRIC = "top1_accuracy"
N_EXAMPLES = 10000

TASK = f"""Reproduce the published top-1 accuracy (in percent) of the mmpretrain
model-zoo ResNet-18 (b16x8, CIFAR-10) on the CIFAR-10 test set
({N_EXAMPLES} images).

The mmpretrain repository is checked out directly in the working directory (its
`tools/`, `configs/`, etc. are at the working-directory root), the trained
checkpoint is at `ckpt.pth`, and the CIFAR-10 data is provisioned under
`data/cifar10/`. The environment is CPU-only and offline. Navigate the repository
to find the evaluation entry point and the matching ResNet-18 CIFAR-10 config,
run the evaluation against the checkpoint, and report the top-1 accuracy as a
percentage."""

EVIDENCE = f"""The eval must WRITE a file `predictions.json` in the working
directory: a JSON list of exactly {N_EXAMPLES} integers — the model's predicted
class id for each CIFAR-10 test image, in test order. The repository's own
`tools/test.py` can dump per-sample predictions (e.g. via `--out`); load that dump
and write `predictions.json`. An external verifier recomputes top-1 accuracy from
this file against held-out gold labels it loads itself; it ignores anything you
print. Do NOT hardcode predictions — only the model's real per-sample predictions
reproduce the target."""

PUBLIC_RESULT_PROTOCOL = f"""Write `predictions.json` in the working directory:
a JSON list of exactly {N_EXAMPLES} integers, containing the model's predicted
class id for each CIFAR-10 test image in test order. The external verifier
recomputes top-1 accuracy from this file and ignores printed aggregate metrics.
Generate the list from real model inference; do not hardcode predictions."""

def _recompute(workdir: Path):
    """Verifier-side top-1 from the agent's per-sample predictions vs pinned gold."""
    pred_path = workdir / "predictions.json"
    if not pred_path.is_file():
        return None
    try:
        preds = json.loads(pred_path.read_text())
        gold = json.loads(GOLD_LABELS.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(preds, list) or len(preds) != len(gold):
        return None
    try:
        correct = sum(int(p) == int(g) for p, g in zip(preds, gold))
    except (TypeError, ValueError):
        return None
    return (100.0 * correct / len(gold), len(gold))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def _make_public_contract_diagnostics(workdir: Path, n_examples: int):
    def _public_contract_diagnostics(session) -> list[str]:
        if not (workdir / "predictions.json").is_file():
            issue = (
                f"No `predictions.json` written. Run the repo's test tool with "
                f"per-sample prediction dumping and write a JSON list of "
                f"{n_examples} predicted class ids in test order."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1500:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        rec = _recompute(workdir)
        if rec is None:
            return [
                f"`predictions.json` is malformed or not a list of {n_examples} "
                f"integer class ids."
            ]
        acc, _ = rec
        if not 0.0 <= acc <= 100.0:
            return ["recomputed top-1 must be a percentage in 0-100."]
        return []

    return _public_contract_diagnostics


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _make_copy_clean_source(workdir: Path):
    def _copy_clean_source() -> None:
        # Repo CONTENTS at the workdir root — same convention as the OpenOOD and
        # RobustBench oracles, and what `git clone && cd` actually gives you:
        # tools/test.py, configs/, data/, ckpt.pth all at one level.
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(
            SOURCE,
            workdir,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        # The checkpoint, at the root the config's run cwd sees.
        shutil.copy2(CKPT_SOURCE, workdir / "ckpt.pth")
        # CIFAR-10 data (extracted batches + tarball) where the config expects it
        # relative to the run cwd: data/cifar10/.
        data_dst = workdir / "data" / "cifar10"
        data_dst.mkdir(parents=True, exist_ok=True)
        for child in CIFAR_SOURCE.iterdir():
            if child.is_dir():
                shutil.copytree(child, data_dst / child.name, dirs_exist_ok=True)
            else:
                shutil.copy2(child, data_dst / child.name)

    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path):
    def _assert_blind_workspace() -> None:
        target = f"{EXPECTED:.2f}"  # "94.82"
        # Only scan small text files the agent could read; skip the repo's own
        # docs/changelogs (large, and the published number may appear in model-zoo
        # tables that are part of the public repo, not a private leak).
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
                continue
            try:
                if target in path.read_text(errors="replace"):
                    raise RuntimeError(
                        f"private target leaked into blind workspace: {path}"
                    )
            except OSError:
                continue

    return _assert_blind_workspace


def _make_execute_eval():
    def _execute_eval(session: DockerSession):
        syntax = session.shell("python -m py_compile eval_mmpretrain.py", timeout=120)
        if not syntax.ok:
            return syntax
        return session.shell("python eval_mmpretrain.py", timeout=1800)

    return _execute_eval


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "mmpretrain_resnet18_multi_rag" / attempt
    artifact_dir = ROOT / "evals" / "runs" / f"mmpretrain_resnet18_multi_rag_{attempt}"

    contract_diagnostics = _make_public_contract_diagnostics(workdir, N_EXAMPLES)

    return OracleConfig(
        name="mmpretrain_resnet18",
        task=TASK,
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_mmpretrain.py",
        make_session=lambda: DockerSession(
            workdir, image=IMAGE, mem="6g", cpus=6.0, default_timeout=1800
        ),
        session_go_offline=True,
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(),
        public_contract_passes=lambda session: not contract_diagnostics(session),
        chance_level=10.0,  # CIFAR-10 top-1 (10 balanced classes)
        verify_kwargs={"expected_num_examples": N_EXAMPLES, "recompute_fn": _recompute},
        public_result_protocol=PUBLIC_RESULT_PROTOCOL,
        public_execution_command="python eval_mmpretrain.py",
        search_extra_exclude={
            "eval_mmpretrain.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
