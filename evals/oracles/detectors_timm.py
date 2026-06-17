"""Parameterized `detectors`-library timm-registration oracle (multi-RAG).

A repair-loop-exercising family. The pretrained CIFAR classifiers from the
`detectors` OOD-benchmark library are registered into timm only as a SIDE EFFECT
of ``import detectors`` — a plain ``timm.create_model("resnet18_cifar100",
pretrained=True)`` raises ``Unknown model`` and crashes. The fix is one line, but
it is non-obvious: the agent must discover it from the provisioned model card.

This makes a clean repair arc: first attempt (naive timm) → runtime crash → the
deterministic contract reports the blocking error → a Repair role searches the
model card, finds ``import detectors``, and fixes it. CIFAR-100 adds a second
trap (the HF split exposes ``fine_label``/``coarse_label``, not ``label``).

The published accuracy is SCRUBBED from the provisioned model card (the blind
target is never shown); the agent must still run the real eval to produce it.
Model weights + dataset are pre-cached, so the eval runs CPU-only and offline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.types import OracleConfig
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
ORACLE_VENV = ROOT / ".venv-oracle"  # has timm + detectors + datasets + torch
CARDS_DIR = Path(__file__).resolve().parent / "detectors_cards"
GOLD_DIR = Path(__file__).resolve().parent / "gold"

METRIC = "top1_accuracy"

def _make_recompute(gold_path: Path):
    """Verifier-side metric: score the agent's per-sample predictions against the
    pinned gold labels. Returns ``(top1_pct, n)`` or ``None``."""
    def _recompute(workdir: Path):
        pred_path = workdir / "predictions.json"
        if not pred_path.is_file():
            return None
        try:
            preds = json.loads(pred_path.read_text())
            gold = json.loads(gold_path.read_text())
        except (ValueError, OSError):
            return None
        if not isinstance(preds, list) or len(preds) != len(gold):
            return None
        try:
            correct = sum(int(p) == int(g) for p, g in zip(preds, gold))
        except (TypeError, ValueError):
            return None
        return (100.0 * correct / len(gold), len(gold))

    return _recompute


def _make_public_contract_diagnostics(workdir: Path, recompute, num_examples: int, num_classes: int):
    chance = 100.0 / num_classes

    def _public_contract_diagnostics(session) -> list[str]:
        if not (workdir / "predictions.json").is_file():
            issue = (
                f"No `predictions.json` written. The eval must write a JSON list of "
                f"{num_examples} predicted class ids in dataset order."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1500:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        rec = recompute(workdir)
        if rec is None:
            return [
                f"`predictions.json` is malformed or not a list of exactly "
                f"{num_examples} integer class ids."
            ]
        acc, _ = rec
        if acc <= chance * 1.5:
            return [
                f"Recomputed accuracy ({acc:.2f}) is at/near the {chance:.2f}% "
                f"random-chance baseline for this {num_classes}-class task. Inspect "
                f"model loading, label mapping, preprocessing, and metric computation "
                f"against public source evidence."
            ]
        return []

    return _public_contract_diagnostics


def _scrub_card(text: str, expected: float) -> str:
    """Drop lines that reveal the published number, keep the loading recipe.

    Removes the model-index `value:` line and the human-readable accuracy line
    (both the fraction form, e.g. 0.7926, and the percentage form, e.g. 79.26),
    so the provisioned card teaches `import detectors` without leaking the blind
    target."""
    frac = f"{expected / 100:.4f}".rstrip("0")  # "0.7926"
    pct = f"{expected:.2f}".rstrip("0").rstrip(".")  # "79.26"
    out = []
    for line in text.splitlines():
        low = line.lower()
        if frac in line or pct in line:
            continue
        if "value:" in low and "accuracy" in text[max(0, text.find(line) - 200):text.find(line)].lower():
            continue
        if "test accuracy" in low or "accuracy:" in low:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def _make_copy_clean_source(workdir: Path, model_name: str, expected: float):
    card_src = CARDS_DIR / f"{model_name}.md"

    def _copy_clean_source() -> None:
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)
        raw = card_src.read_text(errors="replace")
        (workdir / "model_card.md").write_text(_scrub_card(raw, expected))

    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path, expected: float):
    targets = (f"{expected:.2f}", f"{expected / 100:.4f}".rstrip("0"))

    def _assert_blind_workspace() -> None:
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".txt", ".json"}:
                continue
            text = path.read_text(errors="replace")
            for t in targets:
                if t in text:
                    raise RuntimeError(
                        f"private target {t!r} leaked into blind workspace: {path}"
                    )

    return _assert_blind_workspace


def _make_execute_eval():
    def _execute_eval(session: Session):
        syntax = session.shell("python -m py_compile eval_detectors.py", timeout=60)
        if not syntax.ok:
            return syntax
        return session.shell(
            "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "python eval_detectors.py",
            timeout=1200,
        )

    return _execute_eval


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(
    *,
    attempt: str,
    model_name: str,
    dataset_desc: str,
    num_examples: int,
    num_classes: int,
    expected: float,
    workspace_slug: str,
    gold_labels: str,
    tolerance: float = 0.10,
) -> OracleConfig:
    workdir = ROOT / "workspaces" / f"{workspace_slug}_multi_rag" / attempt
    artifact_dir = ROOT / "evals" / "runs" / f"{workspace_slug}_multi_rag_{attempt}"
    recompute = _make_recompute(GOLD_DIR / gold_labels)

    evidence = (
        f"The eval must WRITE a file `predictions.json` in the working directory: a "
        f"JSON list of exactly {num_examples} integers — the model's predicted class "
        f"id for each test example, in dataset order. An external verifier recomputes "
        f"top-1 accuracy from this file against held-out gold labels it loads itself; "
        f"it ignores anything you print. Do NOT hardcode the predictions or the "
        f"accuracy — only per-sample predictions from real inference reproduce the target."
    )
    task = (
        f"Reproduce the published top-1 accuracy (in percent) of the pretrained "
        f"model `{model_name}` on {dataset_desc} ({num_examples} examples).\n\n"
        f"A model card for `{model_name}` is provided in the working directory. The "
        f"model loads through timm; the weights and the dataset are pre-cached on "
        f"disk. The environment is CPU-only and offline. Load the model with its "
        f"trained weights and the preprocessing it expects, evaluate on the full "
        f"test set, and report top-1 accuracy as a percentage."
    )
    contract_diagnostics = _make_public_contract_diagnostics(
        workdir, recompute, num_examples, num_classes
    )

    return OracleConfig(
        name=workspace_slug,
        task=task,
        metric=METRIC,
        expected=expected,
        tolerance=tolerance,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_detectors.py",
        make_session=lambda: Session(
            workdir, venv_python=ORACLE_VENV / "bin" / "python", default_timeout=1200
        ),
        session_go_offline=False,
        copy_clean_source=_make_copy_clean_source(workdir, model_name, expected),
        execute_eval=_make_execute_eval(),
        public_contract_passes=lambda session: not contract_diagnostics(session),
        chance_level=100.0 / num_classes,  # balanced top-1 over num_classes
        verify_kwargs={"expected_num_examples": num_examples, "recompute_fn": recompute},
        public_result_protocol=evidence,
        public_execution_command=(
            "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "python eval_detectors.py"
        ),
        search_extra_exclude={
            "eval_detectors.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir, expected),
    )
