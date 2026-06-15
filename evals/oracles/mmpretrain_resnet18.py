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

import ast
import json
import shutil
from pathlib import Path

from agent.multi_rag import OracleConfig, _extract_python
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

# The wrapper must run the repository's test entry AND write the per-sample
# predictions file — it cannot print a literal it never sees (blind).
_REQUIRED_MARKERS = ("predictions.json", "test.py")


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
        raise ValueError(
            "code is missing required public-contract markers "
            f"{missing}: it must run the repository's test entry (tools/test.py) "
            "with per-sample prediction dumping and write predictions.json."
        )
    return code


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
# Role instructions
# ---------------------------------------------------------------------------

NAVIGATOR_INSTRUCTION = f"""You are the Navigator in a collaborative ML
reproduction team working inside a large cloned mmpretrain repository. You
receive no prewritten queries. Formulate your own search_repo queries to locate,
with exact repository paths, the things a correct evaluation needs:
- the evaluation ENTRY POINT: the repo has many look-alike test launchers
  (slurm_test.sh, dist_test.sh, mim launchers) — find the actual Python entry
  `tools/test.py` that runs a single-process CPU test;
- the matching ResNet-18 CIFAR-10 config under `configs/` (the b16x8 cifar10
  variant) and the base configs it inherits (model / dataset / runtime);
- how `tools/test.py` is invoked (config + checkpoint positional args) and what
  metric it prints (the mmengine `accuracy/top1:` line) and on how many images;
- that data is at `data/cifar10/` and the checkpoint at `ckpt.pth`.
Submit a concise grounded handoff with the exact entry path, config path, and the
run command shape. Do not guess or mention the private target.

Task:
{TASK}"""

REPRODUCER_INSTRUCTION = f"""You are the Reproducer/Builder. Generate a complete
CPU-safe `eval_mmpretrain.py` wrapper. You receive a Navigator handoff but no
prewritten RAG queries; search the repo for any remaining uncertainty (exact
config path, the entry's CLI, the metric key) before coding.

Public execution contract:
- run the repository's own evaluation entry `tools/test.py` as a
  subprocess (use `sys.executable`), passing the ResNet-18 CIFAR-10 config and
  the checkpoint `ckpt.pth`; do NOT re-implement the model, dataset, or metric;
- the environment is CPU-only and offline — do not download anything; the data is
  already at `data/cifar10/`;
- make `tools/test.py` DUMP per-sample predictions (e.g. pass `--out results.pkl`,
  which mmengine writes per-sample); then load that dump and extract the predicted
  class id for each test image;
- WRITE `predictions.json`: a JSON list of the {N_EXAMPLES} predicted class ids in
  test order. The verifier recomputes top-1 from this — you need not parse the
  printed accuracy;
- {EVIDENCE}

Do not guess or mention the private target."""

CRITIC_INSTRUCTION = f"""You are an independent Code Critic. Audit the generated
`eval_mmpretrain.py` against the repository. You receive no prewritten queries:
search the highest-risk unverified claim (is `tools/test.py` the right entry? is
the config path correct and the cifar10 b16x8 variant? how does the dump store the
predicted label?) and submit a complete corrected wrapper, not a prose review.

Verify:
- it invokes `tools/test.py` with the correct config + `ckpt.pth`;
- it does not re-implement evaluation or hardcode predictions;
- it dumps per-sample predictions from the tool and writes `predictions.json`:
  a JSON list of {N_EXAMPLES} predicted class ids in test order.
{EVIDENCE}

Do not guess or mention the private target."""

REVIEWER_INSTRUCTION = f"""You are the independent Reviewer. Audit the current
`eval_mmpretrain.py` and the public execution log. Derive a search_repo query
from the concrete execution error or the highest-risk semantic claim. The
deterministic public-contract audit is authoritative. When execution failed,
focus on the latest blocking error (wrong config path, wrong entry, a data path
the tool cannot find, a parse that returned nothing). When execution succeeded,
check:
- the accuracy came from the repository test tool's real output, not a constant;
- it is a percentage (0-100) and num_examples={N_EXAMPLES};
- the correct config + checkpoint were used.
End with exactly `REVIEW_STATUS: PASS` only when no repair is needed; otherwise
end with exactly `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

REPAIR_INSTRUCTION = f"""You are Repair Agent {{round_index}}. Fix the concrete
failure identified by the execution log and the independent Reviewer. Search the
repository for the specific error (e.g. the exact config path, the test entry's
CLI, the metric key, the data root), then submit a corrected complete
`eval_mmpretrain.py`. Preserve all working behavior and the public contract: run
the real `tools/test.py`, parse the printed top-1 accuracy, percentage units,
and a `predictions.json` with {N_EXAMPLES} per-sample predicted class ids in test order, CPU-only offline.
{EVIDENCE}

Do not guess or mention the private target."""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "mmpretrain_resnet18_multi_rag"
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
        validate_code=_validate_code,
        public_contract_passes=lambda session: not contract_diagnostics(session),
        public_contract_diagnostics=contract_diagnostics,
        verify_kwargs={"expected_num_examples": N_EXAMPLES, "recompute_fn": _recompute},
        navigator_instruction=NAVIGATOR_INSTRUCTION,
        reproducer_instruction=REPRODUCER_INSTRUCTION,
        critic_instruction=CRITIC_INSTRUCTION,
        reviewer_instruction=REVIEWER_INSTRUCTION,
        repair_instruction=REPAIR_INSTRUCTION,
        repair_mode_label="full_file_replacement",
        repair_submit_name="submit_code",
        repair_submit_description="Submit the repaired eval_mmpretrain.py.",
        search_extra_exclude={
            "eval_mmpretrain.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
