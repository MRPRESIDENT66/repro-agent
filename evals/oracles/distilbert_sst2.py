"""DistilBERT SST-2 oracle configuration for the multi-RAG orchestration.

NLP domain (HuggingFace transformers). The agent navigates a provisioned
`model_card.md` (label map + dataset facts) and writes a CPU eval that loads the
cached model by name + the GLUE/SST-2 validation split and reports accuracy. Runs
in a local subprocess venv (.venv-oracle, has transformers + datasets); model and
dataset are pre-cached, so the eval runs offline.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
ORACLE_VENV = ROOT / ".venv-oracle"  # has transformers + datasets + torch
# Pinned gold labels the verifier scores against — loaded independently of the
# agent's environment (extracted once from GLUE/SST-2 validation; see
# scripts/extract_gold or the commit that added it).
GOLD_LABELS = ROOT / "evals" / "oracles" / "gold" / "sst2_val_labels.json"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_CACHE_DIR = HF_HUB / f"models--{MODEL_NAME}"

EXPECTED = 91.06
TOLERANCE = 0.10
METRIC = "accuracy"
N_EXAMPLES = 872
CHANCE_LEVEL = 50.0  # binary SST-2; below chance ⇒ inverted label mapping


TASK = f"""Reproduce the published accuracy (in percent) of the HuggingFace model
'{MODEL_NAME}' on the SST-2 validation set (GLUE, {N_EXAMPLES} examples).

A model card (`model_card.md`) is provided in the working directory, and both the
model weights and the GLUE/SST-2 dataset are pre-cached on disk (load the model by
name from the cache). The environment is CPU-only and offline. Load the model with
its own tokenizer, evaluate on the SST-2 validation split, and report top-1
classification accuracy as a percentage."""

EVIDENCE = f"""The eval must WRITE a file `predictions.json` in the working
directory: a JSON list of exactly {N_EXAMPLES} integers — the model's predicted
label (0 or 1) for each SST-2 validation example, in dataset order. An external
verifier recomputes accuracy from this file against held-out gold labels it loads
itself; it ignores anything you print. Do NOT hardcode the predictions or the
accuracy — only the per-sample predictions from real model inference reproduce
the target."""

# Required-construct markers: the script must really load the model + dataset and
# WRITE the per-sample predictions file — it cannot pass by printing a number.
_REQUIRED_MARKERS = ("predictions.json",)
_REQUIRED_USAGE = ("from_pretrained", "load_dataset")


def _recompute(workdir: Path):
    """Verifier-side metric: read the agent's per-sample predictions and score them
    against the pinned gold labels. Returns ``(accuracy_pct, n)`` or ``None`` when
    the predictions are missing / malformed / the wrong count. The agent cannot
    forge this — it would have to make per-sample predictions whose independently
    computed accuracy equals the hidden target."""
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
# Validation
# ---------------------------------------------------------------------------

def _validate_code(content: str) -> str:
    code = _extract_python(content)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"code is not syntactically valid: {exc}") from exc
    missing = [m for m in _REQUIRED_MARKERS if m not in code]
    if missing:
        raise ValueError(f"code is missing required public-contract markers: {missing}")
    missing_use = [m for m in _REQUIRED_USAGE if m not in code]
    if missing_use:
        raise ValueError(
            "code must actually load the model and dataset (missing: "
            f"{missing_use}); it cannot hardcode the result."
        )
    return code


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def _make_public_contract_diagnostics(workdir: Path, n_examples: int):
    def _public_contract_diagnostics(session) -> list[str]:
        # Feedback for the repair loop, recomputed from the predictions file (the
        # agent's OWN accuracy — not the hidden target, which it can compute itself
        # from the dataset anyway).
        if not (workdir / "predictions.json").is_file():
            issue = (
                f"No `predictions.json` written. The eval must write a JSON list of "
                f"{n_examples} predicted labels (0/1) in SST-2 validation order."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1200:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        rec = _recompute(workdir)
        if rec is None:
            return [
                f"`predictions.json` is malformed or not a list of exactly "
                f"{n_examples} integer labels."
            ]
        acc, _ = rec
        if acc < CHANCE_LEVEL:
            return [
                f"Recomputed accuracy ({acc:.2f}) is below the {CHANCE_LEVEL} "
                f"random-chance baseline for binary SST-2 — the label mapping is "
                f"almost certainly inverted. Check id2label (0=negative, 1=positive)."
            ]
        return []

    return _public_contract_diagnostics


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _snapshot_dir() -> Path:
    snaps = sorted((MODEL_CACHE_DIR / "snapshots").glob("*"))
    if not snaps:
        raise RuntimeError(
            f"model snapshot not found under {MODEL_CACHE_DIR}; pre-cache the model"
        )
    return snaps[-1]


def _make_copy_clean_source(workdir: Path):
    def _copy_clean_source() -> None:
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)
        # Provision the model's facts as a PROSE card, not a loadable `model/`
        # directory. A config-only dir (no weights) is a trap: an over-eager
        # rewrite to `from_pretrained("./model")` crashes on missing weights. The
        # eval loads by NAME from the offline cache; the card only documents the
        # label map (read from the cached config) and the dataset.
        import json as _json
        id2label = {}
        cfg = _snapshot_dir() / "config.json"
        if cfg.exists():
            try:
                id2label = _json.loads(cfg.read_text(errors="replace")).get("id2label", {})
            except Exception:
                id2label = {}
        label_lines = "\n".join(f"- `{k}` → {v}" for k, v in id2label.items()) or "- (see config)"
        card = (
            f"# {MODEL_NAME}\n\n"
            f"DistilBERT fine-tuned for binary sentiment classification on SST-2.\n\n"
            f"## Label mapping (id2label)\n\n{label_lines}\n\n"
            f"These align with the SST-2 gold labels (0 = negative, 1 = positive).\n\n"
            f"## Dataset\n\nGLUE / SST-2 validation split, {N_EXAMPLES} examples; "
            f"text field `sentence`, gold integer field `label`.\n"
        )
        (workdir / "model_card.md").write_text(card)

    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path):
    def _assert_blind_workspace() -> None:
        target = f"{EXPECTED:.2f}"  # "91.06"
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".csv",
            }:
                continue
            if target in path.read_text(errors="replace"):
                raise RuntimeError(
                    f"private target leaked into blind workspace: {path}"
                )

    return _assert_blind_workspace


def _make_execute_eval():
    def _execute_eval(session: Session):
        syntax = session.shell("python -m py_compile eval_sst2.py", timeout=60)
        if not syntax.ok:
            return syntax
        return session.shell(
            "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "python eval_sst2.py",
            timeout=900,
        )

    return _execute_eval


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "distilbert_sst2_multi_rag"
    artifact_dir = ROOT / "evals" / "runs" / f"distilbert_sst2_multi_rag_{attempt}"

    contract_diagnostics = _make_public_contract_diagnostics(workdir, N_EXAMPLES)

    return OracleConfig(
        name="distilbert_sst2",
        task=TASK,
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_sst2.py",
        make_session=lambda: Session(
            workdir, venv_python=ORACLE_VENV / "bin" / "python", default_timeout=900
        ),
        session_go_offline=False,
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(),
        public_contract_passes=lambda session: not contract_diagnostics(session),
        chance_level=50.0,  # binary SST-2 sentiment classification
        verify_kwargs={"expected_num_examples": N_EXAMPLES, "recompute_fn": _recompute},
        public_result_protocol=EVIDENCE,
        public_execution_command="python eval_sst2.py",
        search_extra_exclude={
            "eval_sst2.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
