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

import ast
import json
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
ORACLE_VENV = ROOT / ".venv-oracle"  # has timm + detectors + datasets + torch
CARDS_DIR = Path(__file__).resolve().parent / "detectors_cards"
GOLD_DIR = Path(__file__).resolve().parent / "gold"

METRIC = "top1_accuracy"

# The script must really load the model + data + predict and WRITE the per-sample
# predictions file — but must NOT be required to contain `import detectors`, since
# discovering that line is the whole point of the task.
_REQUIRED_MARKERS = ("predictions.json",)
_REQUIRED_USAGE = ("timm", "load_dataset")
_PREDICTION_MARKERS = ("argmax", "logits", ".max(", "topk")


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
            "code must load the model via timm and the dataset via load_dataset "
            f"(missing: {missing_use}); it cannot hardcode the result."
        )
    if not any(m in code for m in _PREDICTION_MARKERS):
        raise ValueError(
            "code must actually predict (argmax/logits/topk over the model output)."
        )
    return code


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
                f"random-chance baseline for this {num_classes}-class task — a "
                f"broken eval (wrong label field, wrong preprocessing, or the model "
                f"loaded without its trained weights). Check the model card + label field."
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
# Role instructions (parameterized by description fields)
# ---------------------------------------------------------------------------

def _instructions(model_name, dataset_desc, num_examples, label_hint, evidence):
    nav = f"""You are the Navigator in a collaborative ML reproduction team. You
receive no prewritten queries. The working directory contains a `model_card.md`
for the pretrained model `{model_name}`. Search it (and any uncertainty you have)
to pin down exactly how this checkpoint is loaded and evaluated, then submit a
concise grounded handoff covering:
- the exact loading mechanism for `{model_name}` (it is a timm-registered model —
  read the model card's usage snippet carefully; the registration may require a
  specific import side-effect, not just `timm.create_model`);
- the preprocessing the model expects (read normalization from the loaded model's
  `pretrained_cfg`, do not assume ImageNet defaults);
- the dataset: {dataset_desc} ({num_examples} examples), {label_hint};
- CPU-only, offline loading from the local cache.
Do not guess or mention the private target."""

    rep = f"""You are the Reproducer/Builder. Generate a complete CPU-safe
`eval_detectors.py`. You receive a Navigator handoff and a `model_card.md` in the
working directory; search them for any loading/eval uncertainty before coding.

Public execution contract:
- load `{model_name}` via timm with `pretrained=True`; follow the model card's
  usage snippet exactly — the architecture is registered by an import side-effect,
  so a plain `timm.create_model(...)` may raise `Unknown model`;
- read the normalization mean/std from the loaded model's `pretrained_cfg`;
- load the dataset with `load_dataset(...)`: {dataset_desc}, {num_examples}
  examples, {label_hint};
- run batched CPU inference, take `logits.argmax(-1)` as the predicted class id;
- WRITE the per-sample predictions to `predictions.json` in the working directory:
  a JSON list of the {num_examples} predicted class ids in dataset order. You do
  NOT need to print or compute the accuracy — an external verifier recomputes it;
- {evidence}

Do not guess or mention the private target."""

    crit = f"""You are an independent Code Critic. Audit the generated
`eval_detectors.py` against the model card. You receive no prewritten queries:
search the highest-risk unverified claim and submit a complete corrected script.

Verify:
- the model actually loads with its trained weights (the timm registration import
  side-effect is present if the card requires it — otherwise loading raises
  `Unknown model` or yields random weights);
- normalization is read from `pretrained_cfg`, not assumed;
- the dataset + label field are correct ({label_hint});
- the eval WRITES `predictions.json`: a JSON list of {num_examples} per-sample
  predicted class ids in dataset order, from real inference (not hardcoded).
{evidence}

Do not guess or mention the private target."""

    rev = f"""You are the independent Reviewer. Audit the current
`eval_detectors.py` and the public execution log. Derive a search_repo query from
the concrete execution error or the highest-risk claim. The deterministic
public-contract audit is authoritative. When execution failed, focus on the
latest blocking error (an `Unknown model` / registration error, a missing label
field, a preprocessing problem). When execution succeeded, check that accuracy is
far above chance and came from the real model + the correct label field.
End with exactly `REVIEW_STATUS: PASS` only when no repair is needed; otherwise
end with exactly `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

    rep_fix = f"""You are Repair Agent {{round_index}}. Fix the concrete failure
identified by the execution log and the independent Reviewer. Search `model_card.md`
and the error for the specific fix — e.g. if loading raised `Unknown model`, the
model card's usage snippet shows the required registration import; if the label
field is wrong, the dataset exposes a different field name. Submit a corrected
complete `eval_detectors.py`. Preserve the public contract: timm load with trained
weights, normalization from `pretrained_cfg`, and a `predictions.json` with
{num_examples} per-sample predicted class ids in dataset order, CPU-only offline.
{evidence}

Do not guess or mention the private target."""

    return nav, rep, crit, rev, rep_fix


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
    label_hint: str,
    workspace_slug: str,
    gold_labels: str,
    tolerance: float = 0.10,
) -> OracleConfig:
    workdir = ROOT / "workspaces" / f"{workspace_slug}_multi_rag"
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
    nav, rep, crit, rev, rep_fix = _instructions(
        model_name, dataset_desc, num_examples, label_hint, evidence
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
        validate_code=_validate_code,
        public_contract_passes=lambda session: not contract_diagnostics(session),
        public_contract_diagnostics=contract_diagnostics,
        chance_level=100.0 / num_classes,  # balanced top-1 over num_classes
        verify_kwargs={"expected_num_examples": num_examples, "recompute_fn": recompute},
        public_result_protocol=evidence,
        public_execution_command=(
            "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "python eval_detectors.py"
        ),
        navigator_instruction=nav,
        reproducer_instruction=rep,
        critic_instruction=crit,
        reviewer_instruction=rev,
        repair_instruction=rep_fix,
        repair_mode_label="full_file_replacement",
        repair_submit_name="submit_code",
        repair_submit_description="Submit the repaired eval_detectors.py.",
        search_extra_exclude={
            "eval_detectors.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir, expected),
    )
