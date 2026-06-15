"""OpenOOD EBO oracle configuration for the multi-RAG orchestration."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

from agent.multi_rag import (
    OracleConfig,
    _apply_code_patch,
    _extract_python,
    _patch_tool,
    _review_requires_repair,
)
from exec.docker_session import DockerSession
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos" / "OpenOOD"
IMAGE = "repro-openood:latest"

EXPECTED = 87.58
TOLERANCE = 0.05
EXPECTED_DATASETS = {"cifar100": 9000, "tin": 7793}
EXPECTED_RUNS = ["s0", "s1", "s2"]
AGGREGATION = "dataset_mean_then_run_mean"
METRIC = "near_ood_auroc"
CHECKPOINT_ROOT = "results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
CHANCE_LEVEL = 50.0

NORMALIZATION_SOURCE_REL = "openood/preprocessors/transform.py"
NORMALIZATION_DICT_VAR = "normalization_dict"
NORMALIZATION_KEY = "cifar10"

_FORBIDDEN_IMPORT_PREFIXES = (
    "openood.evaluation_api",
    "openood.evaluators",
    "openood.postprocessors",
    "openood.utils.config",
)
_FORBIDDEN_CLASS_DEFS = {"ResNet18_32x32", "ImglistDataset"}
_FORBIDDEN_CALL_OR_IMPORT_NAMES = {"TestStandardPreProcessor"}
_FORBIDDEN_USE_NAMES = {"UnsafeLoader"}
_FORBIDDEN_CALL_ARG_MARKERS = ("config.yml", "--checkpoint_root")
_REQUIRED_CONTRACT_MARKERS = ("predictions.json",)

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
# AST-level contract validation
# ---------------------------------------------------------------------------

def _call_arg_constant_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for operand in list(node.args) + [kw.value for kw in node.keywords]:
            for inner in ast.walk(operand):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    ids.add(id(inner))
    return ids


def _forbidden_contract_violations(tree: ast.AST) -> list[str]:
    def module_forbidden(module: str | None) -> bool:
        return bool(module) and any(
            module == p or module.startswith(p + ".") for p in _FORBIDDEN_IMPORT_PREFIXES
        )

    call_arg_ids = _call_arg_constant_ids(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if module_forbidden(node.module):
                violations.append(
                    f"forbidden import 'from {node.module} import ...' at line {node.lineno}"
                )
            for alias in node.names:
                if (
                    alias.name in _FORBIDDEN_CALL_OR_IMPORT_NAMES
                    or alias.name in _FORBIDDEN_USE_NAMES
                ):
                    violations.append(
                        f"forbidden import of '{alias.name}' at line {node.lineno}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if module_forbidden(alias.name):
                    violations.append(
                        f"forbidden import '{alias.name}' at line {node.lineno}"
                    )
        elif isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_CLASS_DEFS:
            violations.append(
                f"forbidden re-implementation 'class {node.name}' at line {node.lineno}"
            )
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name in _FORBIDDEN_CALL_OR_IMPORT_NAMES:
                violations.append(
                    f"forbidden instantiation '{name}(...)' at line {node.lineno}"
                )
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_USE_NAMES:
            violations.append(
                f"forbidden use of '{node.attr}' at line {node.lineno}"
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) in call_arg_ids
        ):
            for marker in _FORBIDDEN_CALL_ARG_MARKERS:
                if marker in node.value:
                    violations.append(
                        f"forbidden call argument {marker!r} at line {node.lineno}"
                    )
    seen: set[str] = set()
    return [v for v in violations if not (v in seen or seen.add(v))]


def _normalization_diagnostics_for_code(code: str, source: Path) -> list[str]:
    if not source.is_file():
        return []
    try:
        source_tree = ast.parse(source.read_text(errors="replace"))
        generated_tree = ast.parse(code)
        reference_dict = next(
            ast.literal_eval(node.value)
            for node in source_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == NORMALIZATION_DICT_VAR
                for t in node.targets
            )
        )
        expected_mean, expected_std = reference_dict[NORMALIZATION_KEY]
    except (StopIteration, KeyError, SyntaxError, ValueError):
        return []

    generated_literals: dict[str, object] = {}
    for node in ast.walk(generated_tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                generated_literals[target.id] = value

    def resolve_literal(node: ast.AST) -> object:
        if isinstance(node, ast.Name) and node.id in generated_literals:
            return generated_literals[node.id]
        return ast.literal_eval(node)

    issues: list[str] = []
    for node in ast.walk(generated_tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            else node.func.id if isinstance(node.func, ast.Name)
            else ""
        )
        if name != "Normalize":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        try:
            mean_node = keywords.get("mean") or node.args[0]
            std_node = keywords.get("std") or node.args[1]
            actual_mean = resolve_literal(mean_node)
            actual_std = resolve_literal(std_node)
        except (IndexError, KeyError, ValueError, TypeError):
            continue
        if list(actual_mean) != list(expected_mean) or list(actual_std) != list(expected_std):
            issue = (
                f"Hardcoded CIFAR-10 normalization mismatch with repository source: "
                f"expected mean={expected_mean}, std={expected_std}; "
                f"got mean={actual_mean}, std={actual_std}."
            )
            if issue not in issues:
                issues.append(issue)
    return issues


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
        "than scoring all of them. Fix the data root / path construction so every "
        "list entry resolves and decodes — do not subset or drop items."
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
        f"baseline for this higher-is-better metric. A published method scoring "
        f"below chance indicates an inverted score or label/decision direction — "
        f"correct the scoring/decision polarity in the implementation so the metric "
        f"exceeds chance; do not simply negate the reported number."
    )


def _diagnostic_change_terms(diagnostics: list[str]) -> set[str]:
    joined = " ".join(diagnostics).lower()
    terms: set[str] = set()
    if "dataset counts mismatch" in joined:
        terms.update({"datasets", "len("})
    if "aggregation mismatch" in joined or "does not match dataset_mean" in joined:
        terms.update({"aggregation", "actual", "run_metrics"})
    if "run names mismatch" in joined or "dataset keys" in joined:
        terms.update({"run_metrics", *EXPECTED_RUNS})
    if "percentage points" in joined:
        terms.update({"actual", "run_metrics", "100"})
    if "normalization mismatch" in joined:
        terms.update({"normalize", "std", "mean"})
    if "not valid strict json" in joined:
        terms.update({"json.dumps", "repro_result"})
    missing = re.findall(r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]", joined)
    if missing:
        terms.update(re.findall(r"[a-z0-9_.]+", Path(missing[-1]).name.lower()))
    return terms


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
                "A short count means the pipeline silently dropped items — fix the "
                "data path so every list entry loads; do not subset."
            )
            transcript = list(getattr(session, "transcript", []) or [])
            malformed += _silent_drop_hint(session, len(transcript) if transcript else None)
            issues = [malformed]
            generated = workdir / "eval_ebo.py"
            if generated.is_file():
                issues.extend(
                    _normalization_diagnostics_for_code(
                        generated.read_text(errors="replace"),
                        workdir / NORMALIZATION_SOURCE_REL,
                    )
                )
            return issues

        issues: list[str] = []
        auroc, _ = rec
        below = _below_chance_diagnostic(auroc)
        if below:
            issues.append(below)
        # The buried preprocessing gotcha (missing resize) still shows up as a code
        # check against the repo's normalization source — useful repair feedback.
        generated = workdir / "eval_ebo.py"
        if generated.is_file():
            issues.extend(
                _normalization_diagnostics_for_code(
                    generated.read_text(errors="replace"),
                    workdir / NORMALIZATION_SOURCE_REL,
                )
            )
        return issues

    return _public_contract_diagnostics


def _make_generic_safe_diagnostics(workdir: Path):
    """Oracle-specific sanity check safe under generic prompt mode: the eval's
    hardcoded normalization disagreeing with the repository's OWN normalization
    source. Derivable from the agent's own code + the repo's own files, never the
    hidden target — just "your constants don't match the repo you were told to
    use". (The below-chance direction check is now framework-level, driven by
    OracleConfig.chance_level, so it is not repeated here.)
    """
    def _generic_safe_diagnostics(session: DockerSession) -> list[str]:
        issues: list[str] = []
        generated = workdir / "eval_ebo.py"
        if generated.is_file():
            issues.extend(
                _normalization_diagnostics_for_code(
                    generated.read_text(errors="replace"),
                    workdir / NORMALIZATION_SOURCE_REL,
                )
            )
        return issues

    return _generic_safe_diagnostics


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
        "OPENOOD_EBO.md",
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
# Code validator (closure over workdir for normalization check)
# ---------------------------------------------------------------------------

def _make_validate_code(workdir: Path):
    norm_source = workdir / NORMALIZATION_SOURCE_REL

    def _validate_code(content: str) -> str:
        code = _extract_python(content)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"code is not syntactically valid: {exc}") from exc
        if not all(marker in code for marker in _REQUIRED_CONTRACT_MARKERS):
            missing = [m for m in _REQUIRED_CONTRACT_MARKERS if m not in code]
            raise ValueError(
                f"code is missing required public-contract markers: {missing}"
            )
        violations = _forbidden_contract_violations(tree)
        if violations:
            raise ValueError(
                "code violates the fixed model/CLI contract: " + "; ".join(violations)
            )
        norm_issues = _normalization_diagnostics_for_code(code, norm_source)
        if norm_issues:
            raise ValueError(norm_issues[0])
        return code

    return _validate_code


# ---------------------------------------------------------------------------
# Repair validator factory (patch mode)
# ---------------------------------------------------------------------------

def _make_repair_make_validator(workdir: Path, validate_code):
    eval_path = workdir / "eval_ebo.py"

    def repair_make_validator(
        diagnostics: list[str],
        protected_blocks: set[str],
        accepted_new_blocks: list[str],
    ):
        change_terms = _diagnostic_change_terms(diagnostics)

        def validator(payload: str) -> str:
            return _apply_code_patch(
                eval_path,
                payload,
                validate_code=validate_code,
                protected_blocks=protected_blocks,
                required_change_terms=change_terms,
                accepted_new_blocks=accepted_new_blocks,
            )

        return validator

    return repair_make_validator


# ---------------------------------------------------------------------------
# Role instructions
# ---------------------------------------------------------------------------

NAVIGATOR_INSTRUCTION = f"""You are the Navigator in a collaborative ML
reproduction team. You receive no prewritten repository queries. Identify the
most important unknowns in the task, formulate your own search_repo query, use
the retrieved source to refine later queries when needed, then submit a concise
grounded handoff. Include exact source paths, EBO and AUROC semantics, data and
preprocessing, checkpoint layout, aggregation, and CPU/dependency risks.
Do not guess or mention the private target.

Task:
{TASK}"""

REPRODUCER_INSTRUCTION = f"""You are the Reproducer/Builder. Generate a complete
CPU-safe `eval_ebo.py`. You receive a Navigator handoff but no prewritten RAG
queries. Before coding, identify an implementation uncertainty and call
search_repo with your own query. Use follow-up searches only when retrieved
source exposes another uncertainty, then submit the complete script.

Public execution contract:
- exact OpenOOD ResNet18_32x32 and official s0/s1/s2 checkpoints;
- official CIFAR-10 preprocessing and benchmark image lists; do not reimplement
  the repository's ImglistDataset;
- implement the small torchvision test transform directly from
  `openood/preprocessors/transform.py`; retrieve the complete transform
  pipeline (Resize, CenterCrop, ToTensor, Normalize) — not just the
  normalization mean/std — before coding; do not parse checkpoint `config.yml`
  files or instantiate `TestStandardPreProcessor`;
- EBO energy sign semantics (OOD scores HIGHER than ID), evaluated over every
  sample (id={_ID_COUNT}, cifar100=9000, tin=7793 — do not subset or drop items);
- WRITE `predictions.json` with the per-sample EBO scores per run (id/cifar100/tin),
  as described below — the verifier recomputes the AUROC, you do not print it;
- accept `--root` and use batched DataLoader CPU inference;
- import the model and dataset only from direct modules such as
  `openood.networks.resnet18_32x32` and
  `openood.datasets.imglist_dataset`; do not import `openood.evaluation_api`,
  `openood.evaluators`, or `openood.postprocessors`, because their package
  initializers pull unrelated optional dependencies;
- implement the small EBO score and AUROC calculation locally from retrieved
  repository semantics;
- {EVIDENCE}

Do not guess or mention the private target."""

CRITIC_INSTRUCTION = f"""You are an independent Code Critic. Audit the
generated evaluation script against repository source. You receive no
prewritten queries: choose a search_repo query targeting the highest-risk
unverified claim in the code, and issue follow-up queries only when evidence
requires them. Submit a complete corrected script, not a prose review.

Verify model import, benchmark paths, preprocessing, EBO/AUROC sign,
`--root`, batched CPU execution, and that the eval WRITES `predictions.json` with
the per-run per-sample EBO scores (id={_ID_COUNT}/cifar100=9000/tin=7793), from
real inference over every sample (not subset).
Treat every hardcoded normalization value and any custom Dataset
implementation as high risk: verify them against repository source and prefer
the official ImglistDataset.
Use a small direct torchvision test transform from repository normalization
source; retrieve the complete transform pipeline from the source file and verify
it includes all steps (Resize, CenterCrop, ToTensor, Normalize) — not just the
normalization values. Reject checkpoint `config.yml` parsing and
`TestStandardPreProcessor`, which add irrelevant serialized-config failure modes.
Allow only direct OpenOOD model/dataset module imports. Reject
`openood.evaluation_api`, `openood.evaluators`, and `openood.postprocessors`;
use a minimal local EBO/AUROC implementation instead.
The script must satisfy:
{EVIDENCE}

Do not guess or mention the private target."""

REVIEWER_INSTRUCTION = """You are the independent Reviewer. Audit the
current implementation and public execution log. You receive no prewritten
queries. Derive a search_repo query from the concrete execution error or the
highest-risk semantic claim in the current code. Use repository evidence to
explain the finding. The deterministic public-contract audit is authoritative:
do not ignore its failures, and do not request changes to behavior already
demonstrated by a successful execution unless repository evidence proves a
semantic mismatch. When execution failed, focus the review on the latest
blocking error; defer unrelated semantic concerns until the program runs.
Check the per-dataset AUROC values in the execution log for anomalies: if one
dataset's AUROC is substantially higher than the others (e.g. one near 98-100
while others are in the 80s), this strongly indicates a preprocessing mismatch
such as a missing image resize step — retrieve the repository transform source
and verify the complete pipeline. Flag this as REPAIR_REQUIRED even when
execution succeeded and the overall number looks plausible.
End with exactly `REVIEW_STATUS: PASS` only when no repair is needed; otherwise
end with exactly `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target."""

REPAIR_INSTRUCTION = f"""You are Repair Agent {{round_index}}. Fix the
concrete failure identified by the execution log and independent Reviewer.
You receive no prewritten queries. Formulate a search_repo query from the
specific error or disputed semantic claim, inspect the retrieved source, then
submit a small structured patch to the current `eval_ebo.py` only. Never patch
OpenOOD repository source or dependency files. Preserve all unrelated working code. Each
patch edit must contain exact existing `old` code that occurs once and its
replacement `new` code; do not submit the complete file.

The deterministic public-contract audit is authoritative. When it lists a
failure, the patch must directly change the code responsible for that failure.
Do not revert code blocks already endorsed by the independent Reviewer; but code
the Reviewer still disputes (e.g. a suspected EBO/AUROC sign) may and should be
changed. Fix only the latest blocking execution error in this round. Submit at
most two small edits; defer unrelated concerns until the next execution result.

Preserve working behavior and the public contract: correct EBO/AUROC direction,
exact per-sample coverage (cifar100=9000, tin=7793), a `predictions.json` with the
per-run id={_ID_COUNT}/cifar100=9000/tin=7793 EBO scores, `--root`, batched
DataLoader CPU inference, and no unrelated broad-package imports.
{EVIDENCE}

Do not guess or mention the private target."""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(attempt: str) -> OracleConfig:
    workdir = ROOT / "workspaces" / "openood_ebo_multi_rag"
    artifact_dir = ROOT / "evals" / "runs" / f"openood_ebo_multi_rag_{attempt}"

    validate_code = _make_validate_code(workdir)
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
        validate_code=validate_code,
        public_contract_passes=public_contract_passes,
        public_contract_diagnostics=contract_diagnostics,
        generic_safe_diagnostics=_make_generic_safe_diagnostics(workdir),
        chance_level=CHANCE_LEVEL,
        verify_kwargs={
            "expected_num_examples": None,
            "recompute_fn": _recompute,
        },
        public_result_protocol=EVIDENCE,
        public_execution_command=f"python eval_ebo.py --root {CHECKPOINT_ROOT}",
        navigator_instruction=NAVIGATOR_INSTRUCTION,
        reproducer_instruction=REPRODUCER_INSTRUCTION,
        critic_instruction=CRITIC_INSTRUCTION,
        reviewer_instruction=REVIEWER_INSTRUCTION,
        repair_instruction=REPAIR_INSTRUCTION,
        repair_mode_label="structured_exact_replacement_patch",
        repair_submit_name="submit_patch",
        repair_submit_description=(
            "Submit small exact-replacement edits for the current eval_ebo.py."
        ),
        repair_submit_schema=_patch_tool(
            "submit_patch",
            "Submit small exact-replacement edits for the current eval_ebo.py.",
            max_items=2,
        ),
        repair_submission_adapter=lambda arguments: json.dumps(arguments),
        repair_synthesis_instruction=(
            "Return only one JSON object with `edits` and `rationale`, "
            "using the submit_patch schema. Every `old` string must come "
            "from the current eval_ebo.py. Do not patch repository source "
            "or return the complete file."
        ),
        repair_make_validator=_make_repair_make_validator(workdir, validate_code),
        make_endorsed=lambda run_ok, contract_passes, review_path: (
            run_ok and contract_passes and not _review_requires_repair(review_path)
        ),
        search_extra_exclude={
            "eval_ebo.py",
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
        },
        assert_blind_workspace=_make_assert_blind_workspace(workdir),
    )
