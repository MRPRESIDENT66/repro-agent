"""RobustBench Carmon2019Unlabeled oracle configuration for the multi-RAG orchestration."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
from exec.session import Session

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "robustbench"
MODEL_DIR = ROOT / "repos" / "robustbench_models"
DATA_DIR = ROOT / "repos" / "robustbench_data"
# RobustBench's load_clean_dataset(CIFAR10, n=50) returns the first 50 test images
# in standard torchvision order — i.e. the first 50 of the CIFAR-10 test set.
GOLD_LABELS = ROOT / "evals" / "oracles" / "gold" / "cifar10_test_labels.json"

EXPECTED = 52.0
TOLERANCE = 2.0
METRIC = "robust_accuracy"

MODEL_NAME = "Carmon2019Unlabeled"
DATASET = "cifar10"
THREAT_MODEL = "Linf"
N_EXAMPLES = 50
EPSILON = 0.031372549
AA_ATTACKS = ["apgd-ce", "apgd-dlr"]
AA_RESTARTS = 1

TASK = f"""Reproduce the robust accuracy of {MODEL_NAME} on {DATASET} under the
{THREAT_MODEL} threat model using AutoAttack (custom version: {AA_ATTACKS},
{AA_RESTARTS} restart each).

The RobustBench repository (fixed commit), the pre-downloaded model checkpoint,
and the CIFAR-10 test data are already present on disk. The environment is
CPU-only and offline. Evaluate on the first {N_EXAMPLES} examples of the test
set with epsilon={EPSILON}."""

EVIDENCE = f"""The eval must WRITE a file `predictions.json` in the working
directory: a JSON list of exactly {N_EXAMPLES} integers — the model's predicted
class id on each ADVERSARIAL example produced by AutoAttack, in dataset order. An
external verifier computes robust accuracy = fraction of adversarial predictions
that equal the true label (which the verifier loads itself). It ignores anything
you print. Do NOT hardcode predictions or the accuracy — only the model's real
predictions on the attacked inputs reproduce the target."""

_REQUIRED_MARKERS = ("predictions.json", "AutoAttack")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_code(content: str) -> str:
    code = _extract_python(content)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"SyntaxError: {exc}") from exc
    for marker in _REQUIRED_MARKERS:
        if marker not in code:
            raise ValueError(f"Missing required marker: {marker!r}")
    return code


def _make_recompute(n_examples: int):
    """Verifier-side robust accuracy: the eval dumps the model's predictions on the
    ADVERSARIAL examples; the verifier scores them against the first-n gold labels
    it loads itself. Returns ``(robust_acc_pct, n)`` or ``None``."""
    def _recompute(workdir: Path):
        pred_path = workdir / "predictions.json"
        if not pred_path.is_file():
            return None
        try:
            preds = json.loads(pred_path.read_text())
            gold = json.loads(GOLD_LABELS.read_text())[:n_examples]
        except (ValueError, OSError):
            return None
        if not isinstance(preds, list) or len(preds) != n_examples:
            return None
        try:
            correct = sum(int(p) == int(g) for p, g in zip(preds, gold))
        except (TypeError, ValueError):
            return None
        return (100.0 * correct / n_examples, n_examples)

    return _recompute


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def _make_public_contract_diagnostics(workdir: Path, recompute, n_examples: int):
    def _public_contract_diagnostics(session) -> list[str]:
        if not (workdir / "predictions.json").is_file():
            issue = (
                f"No `predictions.json` written. After AutoAttack, the eval must "
                f"write a JSON list of {n_examples} predicted class ids — the model's "
                f"prediction on each ADVERSARIAL example, in dataset order."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1500:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        if recompute(workdir) is None:
            return [f"`predictions.json` is malformed or not {n_examples} integer labels."]
        return []

    return _public_contract_diagnostics


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

# The published robust accuracy leaks into the blind workspace: the repo README
# shows a `robust accuracy: 52.00%` worked example. Scrub it (and the bare 52.0
# form) from doc files so the agent cannot read the target — it must still run
# AutoAttack to produce it.
_BLIND_TARGETS = ("52.00", "52.0")
_DOC_SUFFIXES = {".md", ".rst", ".txt", ".ipynb"}
_SCAN_SUFFIXES = _DOC_SUFFIXES | {".py", ".json", ".csv", ".yaml", ".yml"}


def _make_copy_clean_source(workdir: Path):
    def _copy_clean_source() -> None:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(
            SOURCE,
            workdir,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _DOC_SUFFIXES:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if any(t in text for t in _BLIND_TARGETS):
                for t in _BLIND_TARGETS:
                    text = text.replace(t, "[scrubbed]")
                path.write_text(text)
        (workdir / "robustbench_models").symlink_to(MODEL_DIR)
        (workdir / "robustbench_data").symlink_to(DATA_DIR)

    return _copy_clean_source


def _make_assert_blind_workspace(workdir: Path):
    def _assert_blind_workspace() -> None:
        for path in workdir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SCAN_SUFFIXES:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for t in _BLIND_TARGETS:
                if t in text:
                    raise RuntimeError(
                        f"target {t!r} leaked into blind workspace: {path}"
                    )

    return _assert_blind_workspace


def _make_execute_eval(n_examples: int, epsilon: float):
    def _execute_eval(session: Session):
        session.shell("python -m py_compile eval_robustbench.py", timeout=30)
        return session.shell(
            f"PYTHONPATH=. python eval_robustbench.py "
            f"--model_name {MODEL_NAME} "
            f"--model_dir robustbench_models "
            f"--data_dir robustbench_data "
            f"--n_examples {n_examples} "
            f"--epsilon {epsilon}",
            timeout=900,
        )

    return _execute_eval


# ---------------------------------------------------------------------------
# Role instructions
# ---------------------------------------------------------------------------

NAVIGATOR_INSTRUCTION = f"""You are the Navigator in a collaborative ML reproduction team.
Search the RobustBench repository to understand:
- how to call load_model() to load {MODEL_NAME} from a pre-downloaded checkpoint
  (model_dir=robustbench_models; checkpoint is at robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt)
- how to load CIFAR-10 test data from data_dir=robustbench_data (first {N_EXAMPLES} examples)
  and what preprocessing is used
- how AutoAttack 'custom' version works: which attribute sets n_restarts, the
  exact API to run attacks {AA_ATTACKS}
- how robust accuracy is computed and returned (fraction or percentage?)
- CPU-only constraints

Write a concise grounded handoff with exact API calls and file paths.
Task: {TASK}"""

REPRODUCER_INSTRUCTION = f"""You are the Reproducer. Write a complete CPU-safe
`eval_robustbench.py` that:
- loads {MODEL_NAME} via robustbench's load_model() with model_dir=robustbench_models
- loads CIFAR-10 test data (first {N_EXAMPLES} examples) from data_dir=robustbench_data
  using the correct preprocessing from get_preprocessing()
- runs AutoAttack in 'custom' version with attacks_to_run={AA_ATTACKS},
  epsilon={EPSILON}, {AA_RESTARTS} restart per attack, on CPU
- obtains the adversarial examples AutoAttack produces, runs the model on them,
  and takes `argmax` to get the predicted class id for each example
- accepts --model_name, --model_dir, --data_dir, --n_examples, --epsilon CLI args
- WRITES `predictions.json`: a JSON list of the {N_EXAMPLES} predicted class ids on
  the ADVERSARIAL examples, in dataset order (the verifier computes robust accuracy)

Before coding, search for how to set n_restarts on the AutoAttack object.
Use robustbench imports directly; do not reimplement model loading or data loading.
{EVIDENCE}
Do not guess or mention the private target."""

CRITIC_INSTRUCTION = f"""You are an independent Code Critic. Audit the generated
eval_robustbench.py against the RobustBench repository source. Verify:
- load_model() args: model_name, dataset, threat_model, model_dir
- get_preprocessing() is called correctly before load_clean_dataset()
- AutoAttack instantiation: norm='Linf', eps={EPSILON}, version='custom',
  attacks_to_run={AA_ATTACKS}, device=cpu
- n_restarts={AA_RESTARTS} set correctly (check exact attribute path in source)
- robust accuracy is percentage (0–100), not fraction (0–1)
- predictions.json has {N_EXAMPLES} adversarial predictions (not hardcoded)
- CLI args accepted correctly

Submit a complete corrected script, not prose.
{EVIDENCE}
Do not guess or mention the private target."""

REVIEWER_INSTRUCTION = f"""You are the independent Reviewer. Audit the
implementation and execution log. Derive a search_repo query from the concrete
error or highest-risk semantic claim. The deterministic public-contract audit is
authoritative. When execution succeeded, check:
- robust accuracy is percentage (0–100), not fraction
- predictions.json has {N_EXAMPLES} adversarial predictions (not hardcoded)
- AutoAttack was actually run (not skipped) and predictions are on the ADVERSARIAL examples
- predictions.json came from actual model evaluation, not hardcoded
End with exactly `REVIEW_STATUS: PASS` or `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

REPAIR_INSTRUCTION = f"""You are Repair Agent {{round_index}}. Fix the
concrete failure identified by the execution log and Reviewer. Search the repo
for the specific error or API question, then submit a corrected complete script.
Preserve: predictions.json with {N_EXAMPLES} per-sample predictions on the adversarial examples, in dataset order.
{EVIDENCE}
Do not guess or mention the private target."""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "robustbench_multi_rag"
    artifact_dir = ROOT / "evals" / "runs" / f"robustbench_carmon_{attempt}"

    recompute = _make_recompute(N_EXAMPLES)
    contract_diagnostics = _make_public_contract_diagnostics(workdir, recompute, N_EXAMPLES)

    return OracleConfig(
        name="robustbench_carmon",
        task=TASK,
        metric=METRIC,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        attempt=attempt,
        workdir=workdir,
        artifact_dir=artifact_dir,
        eval_script="eval_robustbench.py",
        make_session=lambda: Session(
            workdir,
            venv_python=ROOT / ".venv" / "bin" / "python",
            default_timeout=900,
        ),
        session_go_offline=False,
        copy_clean_source=_make_copy_clean_source(workdir),
        execute_eval=_make_execute_eval(N_EXAMPLES, EPSILON),
        validate_code=_validate_code,
        public_contract_passes=lambda session: not contract_diagnostics(session),
        public_contract_diagnostics=contract_diagnostics,
        verify_kwargs={"expected_num_examples": N_EXAMPLES, "recompute_fn": recompute},
        navigator_instruction=NAVIGATOR_INSTRUCTION,
        reproducer_instruction=REPRODUCER_INSTRUCTION,
        critic_instruction=CRITIC_INSTRUCTION,
        reviewer_instruction=REVIEWER_INSTRUCTION,
        repair_instruction=REPAIR_INSTRUCTION,
        repair_mode_label="full_file_replacement",
        repair_submit_name="submit_code",
        repair_submit_description="Submit the repaired eval_robustbench.py.",
        search_extra_exclude={
            "eval_robustbench.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
