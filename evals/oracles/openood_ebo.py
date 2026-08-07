"""OpenOOD EBO oracle configuration for the multi-RAG orchestration."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from agent.types import OracleConfig
from exec.docker_session import DockerSession
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "OpenOOD"
IMAGE = "repro-openood:latest"
MPS_PYTHON = ROOT / ".venv-oracle" / "bin" / "python"
MPS_SITE = SOURCE / ".mps-site"

EXPECTED = 87.58
TOLERANCE = 0.05
METRIC = "near_ood_auroc"
CHECKPOINT_ROOT = "results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
CHANCE_LEVEL = 50.0

_RUNS = ("s0", "s1", "s2")
_ID_COUNT = 9000  # OpenOOD CIFAR-10 ID *test* split: the 10000-image test set is
# split into 9000 id-test + 1000 id-val; the near-OOD AUROC scores the 9000 id-test.
_OOD = {"cifar100": 9000, "tin": 7793}  # near-OOD sets + their exact sample counts

TASK = """Reproduce the official EBO Near-OOD AUROC for CIFAR-10 using the
official s0, s1, and s2 CrossEntropy ResNet-18 checkpoints and both Near-OOD
datasets, CIFAR-100 and TinyImageNet. The fixed OpenOOD repository, data, and
checkpoints are already present. Preserve repository evaluation semantics and
report percentage AUROC."""


def _execution_backend() -> str:
    backend = os.getenv("OPENOOD_EXECUTION_BACKEND", "docker").strip().lower()
    if backend not in {"docker", "mps"}:
        raise ValueError("OPENOOD_EXECUTION_BACKEND must be 'docker' or 'mps'")
    return backend


def _task_for_backend(backend: str) -> str:
    if backend == "mps":
        device = """
All required assets are local; do not access the network. The host provides Apple
MPS acceleration. Use `REPRO_DEVICE=mps` for both the model and input tensors,
with a CPU fallback only if `torch.backends.mps.is_available()` is false. Do not
reduce sample coverage to improve speed."""
    else:
        device = """
The execution environment is CPU-only and offline."""
    return TASK + device

EVIDENCE = f"""The eval must WRITE a file `predictions.json` in the working
directory: the per-sample EBO energy scores, structured as
{{"s0": {{"id": [{_ID_COUNT} scores for the complete CIFAR-10 ID test set],
         "cifar100": [9000 scores], "tin": [7793 scores]}},
 "s1": {{...}}, "s2": {{...}}}}  (one block per checkpoint).
An external verifier recomputes the Near-OOD AUROC itself (per run, AUROC of each
OOD set vs the ID set; then the dataset mean within each run, then the mean over
runs). It ignores anything you print. Use the EBO energy convention where OOD
samples score HIGHER than ID. Do NOT hardcode scores — only the model's real
per-sample EBO scores reproduce the target."""


def _auc(pos: list[float], neg: list[float]) -> float:
    """AUROC = P(pos > neg) as a percentage (Mann-Whitney, tie-averaged ranks).
    No sklearn dependency, so the verifier runs in the orchestrator venv."""
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda x: x[0])
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0  # 1-based average rank for the tie block
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(ranks[k] for k in range(len(merged)) if merged[k][1] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg) * 100.0


def _recompute(workdir: Path):
    """Verifier-side Near-OOD AUROC from the agent's per-sample EBO scores. Returns
    ``(auroc_pct, n_scored)`` or ``None`` (missing/malformed/wrong-count)."""
    pred_path = workdir / "predictions.json"
    if not pred_path.is_file():
        return None
    try:
        data = json.loads(pred_path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or set(data) != set(_RUNS):
        return None
    run_aucs: list[float] = []
    total = 0
    for run in _RUNS:
        rd = data.get(run)
        if not isinstance(rd, dict):
            return None
        ids = rd.get("id")
        if not isinstance(ids, list) or len(ids) != _ID_COUNT:
            return None
        try:
            id_scores = [float(x) for x in ids]
        except (TypeError, ValueError):
            return None
        ds_aucs: list[float] = []
        for ood, n in _OOD.items():
            scores = rd.get(ood)
            if not isinstance(scores, list) or len(scores) != n:
                return None
            try:
                ds_aucs.append(_auc([float(x) for x in scores], id_scores))
            except (TypeError, ValueError):
                return None
            total += n
        run_aucs.append(sum(ds_aucs) / len(ds_aucs))
    return (sum(run_aucs) / len(run_aucs), total)


def _public_check(workdir: Path) -> bool:
    """Check score shape and types without computing the private target metric."""
    try:
        data = json.loads((workdir / "predictions.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or set(data) != set(_RUNS):
        return False
    expected_counts = {"id": _ID_COUNT, **_OOD}
    for run in _RUNS:
        block = data.get(run)
        if not isinstance(block, dict) or set(block) != set(expected_counts):
            return False
        for name, count in expected_counts.items():
            values = block.get(name)
            if not isinstance(values, list) or len(values) != count:
                return False
            try:
                numbers = [float(value) for value in values]
            except (TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in numbers):
                return False
    return True

# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _make_copy_clean_source(workdir: Path):
    def _copy_clean_source() -> None:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(
            SOURCE,
            workdir,
            ignore=shutil.ignore_patterns(
                ".git",
                ".mps-site",
                "__pycache__",
                "run_nearood_ebo_cpu.py",
                "nearood_ebo_cpu_results.json",
            ),
        )
    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path):
    forbidden_names = {
        "run_nearood_ebo_cpu.py",
        "nearood_ebo_cpu_results.json",
    }

    def _assert_blind_workspace() -> None:
        present = {p.name for p in workdir.rglob("*") if p.is_file()}
        leaked_names = forbidden_names & present
        if leaked_names:
            raise RuntimeError(
                f"private files leaked into blind workspace: {leaked_names}"
            )
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".csv", ".sh",
            }:
                continue
            if "87.58" in path.read_text(errors="replace"):
                raise RuntimeError(
                    f"private target leaked into blind workspace: {path}"
                )

    return _assert_blind_workspace


def _make_execute_eval(workdir: Path, backend: str):
    def _execute_eval(session):
        syntax = session.shell("python -m py_compile eval_ebo.py", timeout=120)
        if not syntax.ok:
            return syntax
        device = "REPRO_DEVICE=mps " if backend == "mps" else ""
        return session.shell(f"{device}python eval_ebo.py --root {CHECKPOINT_ROOT}")
    return _execute_eval


def _make_session(workdir: Path, backend: str):
    if backend == "docker":
        return DockerSession(
            workdir, image=IMAGE, mem="6g", cpus=6.0, default_timeout=1800
        )
    if not MPS_PYTHON.is_file():
        raise RuntimeError(f"OpenOOD MPS Python is missing: {MPS_PYTHON}")
    if not (MPS_SITE / "numpy" / "__init__.py").is_file():
        raise RuntimeError(
            "OpenOOD MPS compatibility packages are missing. Run: "
            f"{MPS_PYTHON} -m pip install --target {MPS_SITE} numpy==1.26.4"
        )
    return Session(
        workdir,
        venv_python=MPS_PYTHON,
        default_timeout=1800,
        extra_env={
            "PYTHONPATH": str(MPS_SITE),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "REPRO_DEVICE": "mps",
        },
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    backend = _execution_backend()
    workdir = ROOT / "workspaces" / "openood_ebo_multi_rag" / attempt
    artifact_dir = ROOT / "evals" / "runs" / f"openood_ebo_multi_rag_{attempt}"

    return OracleConfig(
        name="openood_ebo",
        task=_task_for_backend(backend),
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        expected_num_examples=None,
        recompute_fn=_recompute,
        public_check_fn=_public_check,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_ebo.py",
        make_session=lambda: _make_session(workdir, backend),
        session_go_offline=backend == "docker",
        execution_backend=backend,
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(workdir, backend),
        chance_level=CHANCE_LEVEL,
        public_result_protocol=EVIDENCE,
        public_execution_command=(
            ("REPRO_DEVICE=mps " if backend == "mps" else "")
            + f"python eval_ebo.py --root {CHECKPOINT_ROOT}"
        ),
        search_extra_exclude={
            "eval_ebo.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
