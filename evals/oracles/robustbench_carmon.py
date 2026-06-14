"""RobustBench Carmon2019Unlabeled oracle configuration for the multi-RAG orchestration."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
from exec.session import Session
from verify.check import extract_structured_evidence

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "robustbench"
MODEL_DIR = ROOT / "repos" / "robustbench_models"
DATA_DIR = ROOT / "repos" / "robustbench_data"

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

EVIDENCE = f"""A result counts only when an EXECUTED evaluation command prints:
REPRO_RESULT {{"metric":"{METRIC}","actual":<number>,"num_examples":{N_EXAMPLES}}}
The evaluation program itself must print the line. Do not echo or printf it.
`actual` is robust accuracy in percentage points (0–100)."""

_REQUIRED_MARKERS = ("REPRO_RESULT", "json.dumps")


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


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def _make_public_contract_diagnostics(n_examples: int):
    def _public_contract_diagnostics(session) -> list[str]:
        evidence = extract_structured_evidence(
            session.transcript,
            metric=METRIC,
            expected_num_examples=n_examples,
        )
        if evidence is None:
            return [
                f"No valid REPRO_RESULT line found. Need "
                f'metric="{METRIC}" and num_examples={n_examples} '
                f"in stdout of a successful command."
            ]
        return []

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
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        (workdir / "robustbench_models").symlink_to(MODEL_DIR)
        (workdir / "robustbench_data").symlink_to(DATA_DIR)

    return _copy_clean_source


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
- computes robust accuracy as percentage points (0–100), NOT fraction
- accepts --model_name, --model_dir, --data_dir, --n_examples, --epsilon CLI args
- prints exactly one line: REPRO_RESULT {{"metric":"{METRIC}","actual":<pct>,"num_examples":{N_EXAMPLES}}}
  using json.dumps

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
- num_examples={N_EXAMPLES} in REPRO_RESULT
- CLI args accepted correctly

Submit a complete corrected script, not prose.
{EVIDENCE}
Do not guess or mention the private target."""

REVIEWER_INSTRUCTION = f"""You are the independent Reviewer. Audit the
implementation and execution log. Derive a search_repo query from the concrete
error or highest-risk semantic claim. The deterministic public-contract audit is
authoritative. When execution succeeded, check:
- robust accuracy is percentage (0–100), not fraction
- num_examples={N_EXAMPLES} in REPRO_RESULT
- AutoAttack was actually run (not skipped)
- the result came from actual model evaluation
End with exactly `REVIEW_STATUS: PASS` or `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

REPAIR_INSTRUCTION = f"""You are Repair Agent {{round_index}}. Fix the
concrete failure identified by the execution log and Reviewer. Search the repo
for the specific error or API question, then submit a corrected complete script.
Preserve: percentage accuracy, num_examples={N_EXAMPLES}, REPRO_RESULT format.
{EVIDENCE}
Do not guess or mention the private target."""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "robustbench_multi_rag"
    artifact_dir = ROOT / "evals" / "runs" / f"robustbench_carmon_{attempt}"

    contract_diagnostics = _make_public_contract_diagnostics(N_EXAMPLES)

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
        verify_kwargs={"expected_num_examples": N_EXAMPLES},
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
    )
