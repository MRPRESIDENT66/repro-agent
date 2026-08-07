"""Held-out Sentence-Transformers STS-B oracle.

The generic agent is frozen at commit 8fa152e. This module only provisions a
new public task and keeps the reference scores and metric recomputation on the
oracle side.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from agent.types import OracleConfig
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "sentence-transformers"
PUBLIC_PAIRS = ROOT / "data" / "sentence_transformers_stsb" / "test_pairs.jsonl"
GOLD_SCORES = ROOT / "evals" / "oracles" / "gold" / "stsb_test_scores.json"
ORACLE_VENV = ROOT / ".venv-oracle"

SOURCE_VERSION = "v5.7.0"
SOURCE_COMMIT = "b2a9529cf6312d2b2a8ffa2b64d82fabc1571bd8"
DATASET_ID = "sentence-transformers/stsb"
DATASET_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-mpnet-base-v2"
    / "snapshots"
    / MODEL_REVISION
)

FROZEN_AGENT_COMMIT = "8fa152e"
N_EXAMPLES = 1379
METRIC = "spearman_cosine"
EXPECTED = 0.8342216679474351
TOLERANCE = 0.001

TASK = f"""Evaluate the pinned `{MODEL_ID}` checkpoint on the STS Benchmark test
split using the provisioned Sentence-Transformers {SOURCE_VERSION} repository.

The environment is offline. The model is available at `model/`, and
`stsb_pairs.jsonl` contains {N_EXAMPLES} records with `id`, `sentence1`, and
`sentence2` but no gold score. Encode each sentence with the checkpoint's normal
Sentence-Transformers pipeline and produce paired cosine similarities. An
external verifier will compute Spearman rank correlation against held-out STS-B
scores. Do not train or modify the checkpoint."""

EVIDENCE = f"""The eval must WRITE predictions.json as a JSON list of exactly
{N_EXAMPLES} objects in input order. Every object must have exactly an integer
`id` and a finite numeric `similarity`, with ids 0 through {N_EXAMPLES - 1}.
`similarity` must be the cosine similarity produced by real inference for that
sentence pair. The verifier ignores printed aggregate metrics and independently
recomputes Spearman correlation from these similarities and private gold scores."""


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have the same non-trivial length")
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    numerator = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(math.fsum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(math.fsum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("Spearman input is constant")
    return numerator / (x_norm * y_norm)


def _load_similarities(workdir: Path) -> list[float] | None:
    path = workdir / "predictions.json"
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or len(rows) != N_EXAMPLES:
        return None
    similarities: list[float] = []
    for expected_id, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "similarity"}:
            return None
        if type(row["id"]) is not int or row["id"] != expected_id:
            return None
        try:
            value = float(row["similarity"])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
            return None
        similarities.append(value)
    return similarities


def _recompute(workdir: Path):
    similarities = _load_similarities(workdir)
    if similarities is None:
        return None
    try:
        gold = [float(value) for value in json.loads(GOLD_SCORES.read_text())]
        score = _spearman(gold, similarities)
    except (OSError, TypeError, ValueError):
        return None
    if len(gold) != N_EXAMPLES:
        return None
    return score, N_EXAMPLES


def _make_public_contract_diagnostics(workdir: Path):
    def _public_contract_diagnostics(session) -> list[str]:
        if not (workdir / "predictions.json").is_file():
            issue = (
                f"No predictions.json written. Write {N_EXAMPLES} ordered "
                "objects with integer `id` and finite cosine `similarity`."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1500:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        if _load_similarities(workdir) is None:
            return [
                f"predictions.json must contain exactly {N_EXAMPLES} ordered "
                "objects with ids 0..1378 and finite cosine similarities."
            ]
        return []

    return _public_contract_diagnostics


def _check_assets() -> None:
    required = [SOURCE / "sentence_transformers", PUBLIC_PAIRS, GOLD_SCORES]
    missing = [str(path) for path in required if not path.exists()]
    if not (MODEL_SNAPSHOT / "model.safetensors").exists():
        missing.append(str(MODEL_SNAPSHOT / "model.safetensors"))
    if missing:
        raise RuntimeError(
            "missing STS-B holdout assets; run "
            "`python scripts/prepare_sentence_transformers_stsb.py`: "
            + ", ".join(missing)
        )


def _make_copy_clean_source(workdir: Path):
    def _copy_clean_source() -> None:
        _check_assets()
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(
            SOURCE,
            workdir,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "docs", "tests"
            ),
        )
        shutil.copy2(PUBLIC_PAIRS, workdir / "stsb_pairs.jsonl")
        (workdir / "model").symlink_to(MODEL_SNAPSHOT, target_is_directory=True)
        (workdir / ".home").mkdir()
        provenance = {
            "source_version": SOURCE_VERSION,
            "source_commit": SOURCE_COMMIT,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_split": "test",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "num_examples": N_EXAMPLES,
        }
        (workdir / "benchmark.json").write_text(json.dumps(provenance, indent=2) + "\n")

    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path):
    def _assert_blind_workspace() -> None:
        target_markers = (f"{EXPECTED:.9f}", f"{100 * EXPECTED:.6f}")
        for path in workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name == "stsb_pairs.jsonl":
                for line in path.read_text(errors="replace").splitlines():
                    row = json.loads(line)
                    if set(row) != {"id", "sentence1", "sentence2"}:
                        raise RuntimeError("public STS-B input contains non-public fields")
                continue
            if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".rst"}:
                continue
            text = path.read_text(errors="replace")
            if any(marker in text for marker in target_markers):
                raise RuntimeError(f"private target leaked into blind workspace: {path}")

    return _assert_blind_workspace


def _make_execute_eval():
    command = (
        "PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "TOKENIZERS_PARALLELISM=false python eval_stsb.py "
        "--input stsb_pairs.jsonl --model model --output predictions.json"
    )

    def _execute_eval(session: Session):
        syntax = session.shell("python -m py_compile eval_stsb.py", timeout=60)
        if not syntax.ok:
            return syntax
        return session.shell(command, timeout=900)

    return _execute_eval


def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "sentence_transformers_stsb" / attempt
    artifact_dir = ROOT / "evals" / "runs" / f"sentence_transformers_stsb_{attempt}"
    diagnostics = _make_public_contract_diagnostics(workdir)

    return OracleConfig(
        name="sentence_transformers_stsb",
        task=TASK,
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_stsb.py",
        make_session=lambda: Session(
            workdir,
            venv_python=ORACLE_VENV / "bin" / "python",
            default_timeout=900,
            extra_env={
                "HOME": str(workdir / ".home"),
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
            },
        ),
        session_go_offline=False,
        execution_backend="local-offline-cpu-or-mps",
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(),
        public_contract_passes=lambda session: not diagnostics(session),
        verify_kwargs={"expected_num_examples": N_EXAMPLES, "recompute_fn": _recompute},
        public_result_protocol=EVIDENCE,
        public_execution_command=(
            "PYTHONPATH=. python eval_stsb.py --input stsb_pairs.jsonl "
            "--model model --output predictions.json"
        ),
        search_extra_exclude={
            "eval_stsb.py",
            "predictions.json",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
