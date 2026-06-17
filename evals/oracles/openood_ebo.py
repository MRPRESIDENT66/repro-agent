"""OpenOOD EBO oracle configuration for the multi-RAG orchestration."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from agent.types import OracleConfig
from exec.docker_session import DockerSession
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "OpenOOD"
IMAGE = "repro-openood:latest"

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
checkpoints are already present. The environment is CPU-only and offline.
Preserve repository evaluation semantics and report percentage AUROC."""

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

# ---------------------------------------------------------------------------
# Contract diagnostics
# ---------------------------------------------------------------------------

_DROP_SIGNAL_RE = re.compile(
    r"broken|FileNotFoundError|No such file|cannot identify image|truncat|"
    r"could not|UnidentifiedImageError|skipp",
    re.IGNORECASE,
)


def _silent_drop_hint(session: DockerSession, command_index: int | None) -> str:
    transcript = list(getattr(session, "transcript", []) or [])
    run = None
    if command_index and 1 <= command_index <= len(transcript):
        run = transcript[command_index - 1]
    text = f"{getattr(run, 'stdout', '')}\n{getattr(run, 'stderr', '')}" if run else ""
    dropped = len(_DROP_SIGNAL_RE.findall(text))
    hint = (
        " A short count means the pipeline silently dropped listed items rather "
        "than scoring all of them. Inspect data loading, path resolution, and "
        "decode errors against the public task; do not subset or drop items."
    )
    if dropped:
        hint += (
            f" The evaluation log shows at least {dropped} drop/error signal(s) "
            f"(e.g. 'broken' / FileNotFoundError) — those items did not load."
        )
    return hint


def _below_chance_diagnostic(actual: float) -> str | None:
    if actual >= CHANCE_LEVEL:
        return None
    return (
        f"The reported value ({actual}) is below the {CHANCE_LEVEL} random-chance "
        f"baseline for this higher-is-better metric. Inspect the score direction, "
        f"label polarity, and metric aggregation against repository evidence."
    )


def _make_public_contract_diagnostics(workdir: Path):
    def _public_contract_diagnostics(session: DockerSession) -> list[str]:
        # V2: feedback recomputed from the per-sample EBO scores file the eval wrote.
        if not (workdir / "predictions.json").is_file():
            issue = (
                "No `predictions.json` (the per-sample EBO scores file) was written. "
                f"It must be {{s0,s1,s2}} each with `id` ({_ID_COUNT}), "
                "`cifar100` (9000) and `tin` (7793) score lists."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                from agent.multi_rag import _search_evidence, _missing_path_hints
                failure = _search_evidence(f"{latest.stdout}\n{latest.stderr}")
                hints = _missing_path_hints(f"{latest.stdout}\n{latest.stderr}", workdir)
                if failure:
                    issue += f" Fix the latest blocking execution error first:\n{failure}"
                if hints:
                    issue += "\nExisting files beside the missing path:\n" + "\n".join(hints)
            return [issue]

        rec = _recompute(workdir)
        if rec is None:
            malformed = (
                "`predictions.json` is malformed: it must be a dict with keys "
                f"{{s0,s1,s2}}, each a dict with `id` (exactly {_ID_COUNT} scores), "
                "`cifar100` (exactly 9000 scores) and `tin` (exactly 7793 scores). "
                "Inspect the public result protocol and the latest execution log."
            )
            transcript = list(getattr(session, "transcript", []) or [])
            malformed += _silent_drop_hint(session, len(transcript) if transcript else None)
            return [malformed]

        issues: list[str] = []
        auroc, _ = rec
        below = _below_chance_diagnostic(auroc)
        if below:
            issues.append(below)
        return issues

    return _public_contract_diagnostics


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


def _make_execute_eval(workdir: Path):
    def _execute_eval(session: DockerSession):
        syntax = session.shell("python -m py_compile eval_ebo.py", timeout=120)
        if not syntax.ok:
            return syntax
        return session.shell(f"python eval_ebo.py --root {CHECKPOINT_ROOT}")
    return _execute_eval


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "openood_ebo_multi_rag" / attempt
    artifact_dir = ROOT / "evals" / "runs" / f"openood_ebo_multi_rag_{attempt}"

    contract_diagnostics = _make_public_contract_diagnostics(workdir)

    def public_contract_passes(session) -> bool:
        return not contract_diagnostics(session)

    return OracleConfig(
        name="openood_ebo",
        task=TASK,
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_ebo.py",
        make_session=lambda: DockerSession(
            workdir, image=IMAGE, mem="6g", cpus=6.0, default_timeout=1800
        ),
        session_go_offline=True,
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(workdir),
        public_contract_passes=public_contract_passes,
        chance_level=CHANCE_LEVEL,
        verify_kwargs={
            "expected_num_examples": None,
            "recompute_fn": _recompute,
        },
        public_result_protocol=EVIDENCE,
        public_execution_command=f"python eval_ebo.py --root {CHECKPOINT_ROOT}",
        search_extra_exclude={
            "eval_ebo.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
