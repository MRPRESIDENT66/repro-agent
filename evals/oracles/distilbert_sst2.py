"""DistilBERT SST-2 oracle configuration for the multi-RAG orchestration.

NLP domain (HuggingFace transformers). The agent navigates the locally
provisioned model snapshot (config.json label map, tokenizer files) and writes a
CPU eval that loads the cached model + GLUE/SST-2 validation split and reports
accuracy. Runs in a local subprocess venv (.venv-oracle, has transformers +
datasets); model and dataset are pre-cached, so the eval runs offline.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
from exec.session import Session
from verify.check import extract_structured_evidence

ROOT = Path(__file__).resolve().parents[2]
ORACLE_VENV = ROOT / ".venv-oracle"  # has transformers + datasets + torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_CACHE_DIR = HF_HUB / f"models--{MODEL_NAME}"

EXPECTED = 91.06
TOLERANCE = 0.10
METRIC = "accuracy"
N_EXAMPLES = 872
CHANCE_LEVEL = 50.0  # binary SST-2; below chance ⇒ inverted label mapping

# Small snapshot files the agent may navigate (the loadable model — NOT the
# README, which carries the published number and must stay out of the blind WS).
_RAG_FILES = ("config.json", "tokenizer_config.json", "vocab.txt")

TASK = f"""Reproduce the published accuracy (in percent) of the HuggingFace model
'{MODEL_NAME}' on the SST-2 validation set (GLUE, {N_EXAMPLES} examples).

The model snapshot (config + tokenizer) is provisioned in the working directory,
and both the model weights and the GLUE/SST-2 dataset are pre-cached on disk. The
environment is CPU-only and offline. Load the model with its own tokenizer,
evaluate on the SST-2 validation split, and report top-1 classification accuracy
as a percentage."""

EVIDENCE = f"""A result counts only when an EXECUTED evaluation command prints:
REPRO_RESULT {{"metric":"{METRIC}","actual":<number>,"num_examples":{N_EXAMPLES}}}
The evaluation program itself must print the line. Do not echo or printf it.
`actual` is classification accuracy in percentage points (0-100)."""

# Required-construct markers: the script must really load the model + dataset and
# serialize the contract — it cannot just print a literal it never sees (blind).
_REQUIRED_MARKERS = ("REPRO_RESULT", "json.dumps")
_REQUIRED_USAGE = ("from_pretrained", "load_dataset")


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

def _make_public_contract_diagnostics(n_examples: int):
    def _public_contract_diagnostics(session) -> list[str]:
        evidence = extract_structured_evidence(
            session.transcript,
            metric=METRIC,
            expected_num_examples=n_examples,
        )
        if evidence is None:
            issue = (
                f"No valid REPRO_RESULT was produced by a successful evaluation "
                f'command. Need metric="{METRIC}" and num_examples={n_examples} in '
                f"the stdout of a successful run."
            )
            latest = next(
                (run for run in reversed(session.transcript) if not run.ok), None
            )
            if latest is not None:
                tail = f"{latest.stdout}\n{latest.stderr}".strip()[-1200:]
                if tail:
                    issue += f"\nFix the latest blocking execution error first:\n{tail}"
            return [issue]
        issues: list[str] = []
        if not 0.0 <= evidence.actual <= 100.0:
            issues.append("accuracy must be a percentage in the 0-100 scale.")
        if evidence.actual < CHANCE_LEVEL:
            issues.append(
                f"The reported accuracy ({evidence.actual}) is below the "
                f"{CHANCE_LEVEL} random-chance baseline for binary SST-2. This "
                f"almost always means the label mapping is inverted — check the "
                f"model's id2label in config.json against the SST-2 gold labels "
                f"(0=negative, 1=positive); do not simply negate the number."
            )
        return issues

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
        snapshot = _snapshot_dir()
        model_ws = workdir / "model"
        model_ws.mkdir()
        for name in _RAG_FILES:
            src = snapshot / name
            if src.exists():
                shutil.copy2(src, model_ws / name)

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
# Role instructions
# ---------------------------------------------------------------------------

NAVIGATOR_INSTRUCTION = f"""You are the Navigator in a collaborative ML
reproduction team. You receive no prewritten queries. Formulate your own
search_repo query over the provisioned model snapshot (under `model/`) to pin
down the facts a correct evaluation needs, then submit a concise grounded
handoff. Cover:
- the model class to load ({MODEL_NAME}) and that it is a sequence-classification
  head loaded with its own tokenizer;
- the label mapping (inspect `model/config.json` id2label/label2id) and whether
  it aligns with SST-2 gold labels (0=negative, 1=positive);
- the dataset to use: GLUE/SST-2 validation split, {N_EXAMPLES} examples, with
  text field `sentence` and integer field `label`;
- tokenization (the model's own tokenizer, padding + truncation) and CPU-only,
  offline loading from the local cache.
Do not guess or mention the private target.

Task:
{TASK}"""

REPRODUCER_INSTRUCTION = f"""You are the Reproducer/Builder. Generate a complete
CPU-safe `eval_sst2.py`. You receive a Navigator handoff but no prewritten RAG
queries; search the model snapshot for any remaining uncertainty (e.g. the exact
id2label) before coding.

Public execution contract:
- load the model with `AutoModelForSequenceClassification.from_pretrained(
  "{MODEL_NAME}")` and the matching `AutoTokenizer`; the weights are cached, so
  load by name (offline) — set `model.eval()`;
- load the data with `load_dataset(...)` for GLUE/SST-2 validation
  ({N_EXAMPLES} examples); the text field is `sentence`, the gold field is
  `label`; the GLUE config needs a namespaced id;
- tokenize with the model's own tokenizer (padding + truncation), run batched CPU
  inference, take `logits.argmax(-1)` and compare against the gold `label`
  (the model's LABEL_0/LABEL_1 already align with SST-2 0/1 — no remap);
- compute accuracy as a percentage over all {N_EXAMPLES} examples;
- print exactly one strict-JSON `REPRO_RESULT` line using `json.dumps`;
- {EVIDENCE}

Do not guess or mention the private target."""

CRITIC_INSTRUCTION = f"""You are an independent Code Critic. Audit the generated
`eval_sst2.py` against the provisioned model snapshot. You receive no prewritten
queries: search the highest-risk unverified claim (label mapping, dataset id,
text/label field names, split, tokenizer) and submit a complete corrected
script, not a prose review.

Verify:
- the model + tokenizer are loaded from `{MODEL_NAME}`;
- the dataset is the GLUE/SST-2 **validation** split with {N_EXAMPLES} examples,
  reading `sentence` and `label`;
- the label mapping matches `model/config.json` and SST-2 gold labels with no
  spurious inversion;
- accuracy is a percentage (0-100) over all examples;
- exactly one strict-JSON `REPRO_RESULT` via `json.dumps`.
{EVIDENCE}

Do not guess or mention the private target."""

REVIEWER_INSTRUCTION = f"""You are the independent Reviewer. Audit the current
`eval_sst2.py` and the public execution log. Derive a search_repo query from the
concrete execution error or the highest-risk semantic claim. The deterministic
public-contract audit is authoritative. When execution succeeded, check:
- accuracy is a percentage (0-100), not a fraction;
- num_examples = {N_EXAMPLES} (the full validation split, not a subset);
- the label mapping is correct — if accuracy is near or below 50%, suspect an
  inverted label/argmax direction and verify against `model/config.json`;
- the result came from real model inference, not a hardcoded constant.
End with exactly `REVIEW_STATUS: PASS` only when no repair is needed; otherwise
end with exactly `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

REPAIR_INSTRUCTION = f"""You are Repair Agent {{round_index}}. Fix the concrete
failure identified by the execution log and the independent Reviewer. Search the
model snapshot for the specific error or disputed claim, then submit a corrected
complete `eval_sst2.py`. Preserve all working behavior and the public contract:
percentage accuracy, num_examples={N_EXAMPLES}, correct label mapping, strict-JSON
`REPRO_RESULT` via `json.dumps`, CPU-only offline loading.
{EVIDENCE}

Do not guess or mention the private target."""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "distilbert_sst2_multi_rag"
    artifact_dir = ROOT / "evals" / "runs" / f"distilbert_sst2_multi_rag_{attempt}"

    contract_diagnostics = _make_public_contract_diagnostics(N_EXAMPLES)

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
        repair_submit_description="Submit the repaired eval_sst2.py.",
        search_extra_exclude={
            "eval_sst2.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
