"""Run one strict-blind collaborative Multi-Agent + RAG OpenOOD experiment.

Isolated LLM contexts collaborate through auditable files:

1. Each role formulates its own ``search_repo`` queries from its current task,
   implementation, or execution error.
2. Navigator writes a grounded handoff; Builder and Critic generate the eval.
3. Reviewer independently audits execution and routes failures to Repair.

The private target and deterministic verifier stay outside the Docker mount.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from agent.llm import ChatLLM
from agent.loop import AgentResult, TOOLS, run_agent
from exec.docker_session import DockerSession
from retrieval.search import relevant_snippet, search_repo
from verify.check import extract_structured_evidence, verify_run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "repos" / "OpenOOD"
WORKDIR = ROOT / "workspaces" / "openood_ebo_multi_rag"
ATTEMPT = os.environ.get("OPENOOD_MULTI_RAG_ATTEMPT", "002")
ARTIFACT_DIR = ROOT / "evals" / "runs" / f"openood_ebo_multi_rag_{ATTEMPT}"
IMAGE = "repro-openood:latest"

EXPECTED = 87.58
TOLERANCE = 0.05
EXPECTED_DATASETS = {"cifar100": 9000, "tin": 7793}
EXPECTED_RUNS = ["s0", "s1", "s2"]
AGGREGATION = "dataset_mean_then_run_mean"
SEARCH_REPO_TOOL = next(
    tool for tool in TOOLS if tool["function"]["name"] == "search_repo"
)

TASK = """Reproduce the official EBO Near-OOD AUROC for CIFAR-10 using the
official s0, s1, and s2 CrossEntropy ResNet-18 checkpoints and both Near-OOD
datasets, CIFAR-100 and TinyImageNet. The fixed OpenOOD repository, data, and
checkpoints are already present. The environment is CPU-only and offline.
Preserve repository evaluation semantics and report percentage AUROC."""

EVIDENCE = """A result counts only when an EXECUTED evaluation command prints:
REPRO_RESULT {"metric":"near_ood_auroc","actual":<number>,
"datasets":{"cifar100":<count>,"tin":<count>},
"run_metrics":{"s0":{"cifar100":<auroc>,"tin":<auroc>},
"s1":{"cifar100":<auroc>,"tin":<auroc>},
"s2":{"cifar100":<auroc>,"tin":<auroc>}},
"aggregation":"dataset_mean_then_run_mean"}
The evaluation program itself must print the line. Do not echo or printf it.
`actual` must equal the dataset mean within each run, then the mean of runs."""


def _copy_clean_source() -> None:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    shutil.copytree(
        SOURCE,
        WORKDIR,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "run_nearood_ebo_cpu.py",
            "nearood_ebo_cpu_results.json",
        ),
    )


def _assert_blind_workspace() -> None:
    forbidden_names = {
        "run_nearood_ebo_cpu.py",
        "nearood_ebo_cpu_results.json",
        "OPENOOD_EBO.md",
    }
    present = {p.name for p in WORKDIR.rglob("*") if p.is_file()}
    leaked_names = forbidden_names & present
    if leaked_names:
        raise RuntimeError(f"private files leaked into blind workspace: {leaked_names}")
    for path in WORKDIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".csv", ".sh",
        }:
            continue
        if "87.58" in path.read_text(errors="replace"):
            raise RuntimeError(f"private target leaked into blind workspace: {path}")


def _clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


def _public_log(session: DockerSession, start: int) -> str:
    parts = []
    for index, run in enumerate(session.transcript[start:], start + 1):
        parts.append(
            f"## Command {index}\n\n```bash\n{run.command}\n```\n\n"
            f"exit={run.exit_code} timed_out={run.timed_out}\n\n"
            f"```text\n{_clip(run.stdout)}\n{_clip(run.stderr)}\n```\n"
        )
    return "\n".join(parts)


def _save_role_transcript(name: str, result: AgentResult) -> None:
    text = "".join(json.dumps(message) + "\n" for message in result.transcript)
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_transcript.jsonl").write_text(text)


def _save_messages(name: str, messages: list[dict]) -> None:
    text = "".join(json.dumps(message) + "\n" for message in messages)
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_transcript.jsonl").write_text(text)


def _combined_usage(*llms: ChatLLM) -> dict:
    usages = [llm.usage.as_dict() for llm in llms]
    return {
        "llm_calls": sum(item["llm_calls"] for item in usages),
        "prompt_tokens": sum(item["prompt_tokens"] for item in usages),
        "cache_hit_tokens": sum(item["cache_hit_tokens"] for item in usages),
        "completion_tokens": sum(item["completion_tokens"] for item in usages),
        "cost_yuan": round(sum(item["cost_yuan"] for item in usages), 4),
    }


def _extract_python(text: str) -> str:
    """Pull the actual eval script out of a model reply — model-agnostic.

    Some models preface the script with prose and an illustrative snippet in a
    separate fence, so the FIRST code block can be the wrong one. Prefer the
    block that prints the result line; otherwise the largest block (the full
    script). Falls back to the whole text when there is no fence."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [b for b in blocks if "REPRO_RESULT" in b] or blocks
    return max(candidates, key=len).strip() + "\n"


def _submit_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Complete artifact content.",
                    },
                },
                "required": ["content"],
            },
        },
    }


def _patch_tool(name: str, description: str, max_items: int = 8) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "Small exact replacements applied in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {
                                    "type": "string",
                                    "description": "Exact existing code occurring once.",
                                },
                                "new": {
                                    "type": "string",
                                    "description": "Replacement code.",
                                },
                            },
                            "required": ["old", "new"],
                        },
                        "minItems": 1,
                        "maxItems": max_items,
                    },
                    "rationale": {
                        "type": "string",
                        "description": "How the edits address the concrete error.",
                    },
                },
                "required": ["edits", "rationale"],
            },
        },
    }


def _validate_report(content: str) -> str:
    content = content.strip()
    if len(content) < 300:
        raise ValueError("report must contain at least 300 characters")
    if "DSML" in content or "tool_calls" in content:
        raise ValueError("report contains tool-call markup instead of a synthesized artifact")
    return content + "\n"


def _validate_review(content: str) -> str:
    content = _validate_report(content)
    matches = re.findall(r"REVIEW_STATUS:\s*(PASS|REPAIR_REQUIRED)", content)
    if not matches:
        raise ValueError("review must end with REVIEW_STATUS: PASS or REPAIR_REQUIRED")
    body = re.sub(
        r"[*`]*REVIEW_STATUS:\s*(?:PASS|REPAIR_REQUIRED)[*`]*\s*$",
        "",
        content.rstrip(),
    ).rstrip()
    return f"{body}\n\nREVIEW_STATUS: {matches[-1]}\n"


# ============================ PER-ORACLE CONTRACT ============================
# Everything below is OpenOOD-specific *configuration*, not general capability.
# The validation/diagnosis MECHANISMS (`_forbidden_contract_violations`,
# `_normalization_diagnostics_for_code`, ...) are oracle-agnostic and read these
# constants; a different oracle would supply its own. Kept declarative and in one
# place so the capability code carries no task literals.
#
# Module-path prefixes whose package initializers pull unrelated optional deps.
_FORBIDDEN_IMPORT_PREFIXES = (
    "openood.evaluation_api",
    "openood.evaluators",
    "openood.postprocessors",
    "openood.utils.config",
)
# Repository classes the script must reuse, never re-implement.
_FORBIDDEN_CLASS_DEFS = {"ResNet18_32x32", "ImglistDataset"}
# Names that must not be imported or instantiated (they drag in serialized-config
# / optional-dependency failure modes).
_FORBIDDEN_CALL_OR_IMPORT_NAMES = {"TestStandardPreProcessor"}
# Names that must not be used as attributes/identifiers (unsafe YAML loading).
_FORBIDDEN_USE_NAMES = {"UnsafeLoader"}
# Literal flags/paths that must not be USED in real code — i.e. passed to a call
# (argparse `add_argument`, `open`, `Config(...)`, a YAML loader, ...). A bare
# mention in a comment, docstring, or plain string is allowed: it is not part of
# the executed contract.
_FORBIDDEN_CALL_ARG_MARKERS = ("config.yml", "--checkpoint_root")
# Where the repository declares its normalization constants, and the dataset key
# the generated code must match (used by the normalization mismatch check).
NORMALIZATION_SOURCE_REL = "openood/preprocessors/transform.py"
NORMALIZATION_DICT_VAR = "normalization_dict"
NORMALIZATION_KEY = "cifar10"
# General contract-format markers every emitted eval must contain. These are the
# evidence FORMAT (a printed strict-JSON REPRO_RESULT), not an implementation
# shape — the oracle-specific model/CLI/loader choices are enforced by execution
# and the blind verifier, not pre-required here.
_REQUIRED_CONTRACT_MARKERS = ("REPRO_RESULT", "json.dumps")
# The metric id the evidence line must carry, and the checkpoint directory the
# harness points the eval at (where the official s0/s1/s2 runs live).
METRIC = "near_ood_auroc"
CHECKPOINT_ROOT = "results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
# Random-chance baseline of a higher-is-better metric (AUROC = 50). Enables a
# general sanity check: a published method scoring below chance is inverted. Set
# to None for metrics with no such baseline.
CHANCE_LEVEL = 50.0
# =============================================================================


def _call_arg_constant_ids(tree: ast.AST) -> set[int]:
    """Identity of every string Constant that is passed as a (positional or
    keyword) argument to some call — the only place a forbidden flag/path literal
    represents real, executed use rather than prose."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        operands = list(node.args) + [kw.value for kw in node.keywords]
        for operand in operands:
            for inner in ast.walk(operand):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    ids.add(id(inner))
    return ids


def _forbidden_contract_violations(tree: ast.AST) -> list[str]:
    """AST-level fixed-contract violations with line numbers.

    Deliberately structural: a name in a comment, docstring, or plain string is
    NOT a violation — only a real import / class definition / instantiation /
    call / attribute use is. (Comments are never in the AST; the two flag/path
    literals only count when actually passed to a call.)
    """

    def module_forbidden(module: str | None) -> bool:
        return bool(module) and any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
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
                if alias.name in _FORBIDDEN_CALL_OR_IMPORT_NAMES or alias.name in _FORBIDDEN_USE_NAMES:
                    violations.append(
                        f"forbidden import of '{alias.name}' at line {node.lineno}"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if module_forbidden(alias.name):
                    violations.append(f"forbidden import '{alias.name}' at line {node.lineno}")
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
                violations.append(f"forbidden instantiation '{name}(...)' at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_USE_NAMES:
            violations.append(f"forbidden use of '{node.attr}' at line {node.lineno}")
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


def _validate_code(content: str) -> str:
    code = _extract_python(content)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"code is not syntactically valid: {exc}") from exc
    if not all(marker in code for marker in _REQUIRED_CONTRACT_MARKERS):
        missing = [marker for marker in _REQUIRED_CONTRACT_MARKERS if marker not in code]
        raise ValueError(f"code is missing required public-contract markers: {missing}")
    violations = _forbidden_contract_violations(tree)
    if violations:
        raise ValueError(
            "code violates the fixed model/CLI contract: " + "; ".join(violations)
        )
    normalization_issues = _normalization_diagnostics_for_code(
        code,
        WORKDIR / NORMALIZATION_SOURCE_REL,
    )
    if normalization_issues:
        raise ValueError(normalization_issues[0])
    return code


def _extract_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = (fenced.group(1) if fenced else text).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"patch must be one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("patch must be a JSON object")
    return value


def _closest_existing_lines(source: str, target: str, window: int = 8) -> str:
    """When a patch's `old` text isn't in the file, surface the real code most
    similar to it (with line numbers) so the next attempt copies an exact
    snippet. A general code-editing aid — no repo-specific knowledge: it anchors
    on the most similar existing line and shows a window around it."""
    source_lines = source.splitlines()
    target_lines = [line for line in target.splitlines() if line.strip()]
    if not source_lines or not target_lines:
        return ""
    anchor = target_lines[0].strip()
    best_index, best_ratio = 0, -1.0
    for index, line in enumerate(source_lines):
        ratio = SequenceMatcher(None, line.strip(), anchor).ratio()
        if ratio > best_ratio:
            best_ratio, best_index = ratio, index
    lo = max(0, best_index - 2)
    hi = min(len(source_lines), best_index + window)
    return "\n".join(f"{i + 1:>4}: {source_lines[i]}" for i in range(lo, hi))


def _apply_code_patch(
    current_path: Path,
    payload: str,
    *,
    protected_blocks: set[str] | None = None,
    required_change_terms: set[str] | None = None,
    accepted_new_blocks: list[str] | None = None,
) -> str:
    current = current_path.read_text(errors="replace")
    patch = _extract_json_object(payload)
    edits = patch.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise ValueError("patch must contain 1-8 exact replacement edits")
    updated = current
    changed_chars = 0
    changed_fragments: list[str] = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            raise ValueError(f"edit {index} must be an object")
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or len(old) < 5:
            raise ValueError(f"edit {index} requires non-trivial old/new strings")
        occurrences = updated.count(old)
        if occurrences == 0:
            nearby = _closest_existing_lines(updated, old)
            hint = (
                f"\nClosest actual code currently in the file:\n{nearby}" if nearby else ""
            )
            raise ValueError(
                f"edit {index} old text was not found in the current file — it is "
                f"stale or paraphrased. Copy an EXACT snippet from the current "
                f"file as `old`.{hint}"
            )
        if occurrences != 1:
            raise ValueError(
                f"edit {index} old text must occur exactly once; found {occurrences}"
            )
        if old == new:
            raise ValueError(f"edit {index} is a no-op")
        changed_chars += len(old)
        for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
            None, old, new
        ).get_opcodes():
            if tag != "equal":
                changed_fragments.extend(
                    (old[old_start:old_end], new[new_start:new_end])
                )
        updated = updated.replace(old, new, 1)
    if changed_chars > len(current) * 0.65:
        raise ValueError("patch replaces too much of the current file")
    if SequenceMatcher(None, current, updated).ratio() < 0.55:
        raise ValueError("patch does not preserve enough working code")
    for block in protected_blocks or set():
        if block in current and block not in updated:
            raise ValueError("patch changes code already confirmed by a reviewer-endorsed execution")
    if required_change_terms:
        changed = "\n".join(changed_fragments).lower()
        if not any(term.lower() in changed for term in required_change_terms):
            terms = sorted(required_change_terms)
            raise ValueError(
                f"patch does not address deterministic public-contract issue: {terms}"
            )
    if accepted_new_blocks is not None:
        accepted_new_blocks.extend(
            edit["new"] for edit in edits
            if isinstance(edit.get("new"), str) and len(edit["new"]) >= 12
        )
    return _validate_code(updated)


def _search_evidence(context: str) -> str:
    traceback_paths = re.findall(r'File "/workspace/([^"]+\.py)"', context)
    # Any repo-relative source path mentioned in the error context (oracle-agnostic
    # — a directory-qualified .py token, not a specific project's folder names).
    mentioned_paths = re.findall(r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]+\.py)\b", context)
    failures = re.findall(
        r"(?m)^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|"
        r"ModuleNotFoundError|RuntimeError|TypeError|ValueError):.*$",
        context,
    )
    paths = traceback_paths[-4:] or mentioned_paths[-3:]
    unique = list(dict.fromkeys(paths + failures[-1:]))
    return "\n".join(unique)[-2400:]


def _missing_path_hints(context: str) -> list[str]:
    matches = re.findall(
        r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]",
        context,
    )
    if not matches:
        return []
    missing = matches[-1]
    relative = missing.removeprefix("/workspace/").lstrip("./")
    parent = (WORKDIR / relative).parent
    if not parent.is_dir():
        # The parent directory itself doesn't exist — typically a wrong or
        # duplicated data root (e.g. data/images/cifar10/cifar10/...). Walk up to
        # the nearest real ancestor and list its actual contents, so the agent
        # corrects the root from on-disk truth instead of guessing again.
        ancestor = parent
        while not ancestor.is_dir() and WORKDIR in ancestor.parents:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            return []
        try:
            rel = ancestor.relative_to(WORKDIR)
        except ValueError:
            return []
        prefix = "" if str(rel) == "." else f"{rel}/"
        entries = sorted(
            f"{prefix}{child.name}" + ("/" if child.is_dir() else "")
            for child in ancestor.iterdir()
        )
        return entries[:8]
    stem_tokens = set(re.findall(r"[a-z0-9]+", Path(relative).stem.lower()))
    candidates = []
    for path in parent.iterdir():
        if not path.is_file():
            continue
        tokens = set(re.findall(r"[a-z0-9]+", path.stem.lower()))
        overlap = len(stem_tokens & tokens)
        candidates.append((overlap, path.name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    prefix = str(Path(relative).parent)
    return [f"{prefix}/{name}" for _, name in candidates[:8]]


def _search_with_snippets(
    query: str,
    llm: ChatLLM,
    max_files: int = 4,
    context: str | None = None,
) -> str:
    generated = {
        "eval_ebo.py",
        "navigator_report.md",
        "review_report.md",
        "reproducer_public_log.txt",
    }
    generated.update(path.name for path in WORKDIR.glob("*_rag_trace.md"))
    generated.update(path.name for path in WORKDIR.glob("*_transcript.jsonl"))
    ranking_evidence = _search_evidence(context or "")
    path_hints = _missing_path_hints(context or "")
    if path_hints:
        ranking_evidence += (
            "\nExisting files beside the missing path:\n" + "\n".join(path_hints)
        )
    result = search_repo(
        query,
        WORKDIR,
        llm,
        exclude_paths=generated,
        context=ranking_evidence or None,
    )
    paths = []
    for line in result.splitlines():
        match = re.match(r"^\s{2}(\S+)\s+\u2014", line)
        if match and match.group(1) not in paths:
            paths.append(match.group(1))
    snippets = []
    for relative in paths[:max_files]:
        path = WORKDIR / relative
        if path.is_file():
            snippet_query = f"{query}\n{ranking_evidence}"
            snippets.append(
                f"\n## Source: {relative}\n\n"
                f"{relevant_snippet(path, snippet_query, 3200)}"
            )
    evidence_section = (
        f"\n\nError evidence used for ranking:\n{ranking_evidence}"
        if ranking_evidence
        else ""
    )
    return (
        result
        + evidence_section
        + "\n\nRetrieved source snippets:\n"
        + "\n".join(snippets)
    )


def _dynamic_rag_role(
    *,
    name: str,
    session: DockerSession,
    instruction: str,
    context: str,
    output_path: Path,
    submit_name: str,
    submit_description: str,
    validator: Callable[[str], str],
    trigger: str,
    max_steps: int = 7,
    max_queries: int = 3,
    submit_schema: dict | None = None,
    submission_adapter: Callable[[dict], str] | None = None,
    synthesis_instruction: str | None = None,
    synthesis_attempts: int = 3,
) -> tuple[dict, dict]:
    role_llm = ChatLLM()
    rag_llm = ChatLLM()
    synthesis_llm = ChatLLM()
    queries: list[str] = []
    submitted = False
    trace_sections = [
        f"# {name} dynamic RAG trace",
        "",
        f"Trigger: {trigger}",
        "",
        "Queries below were generated by the role at runtime.",
    ]
    trace_path = WORKDIR / f"{name}_rag_trace.md"
    submission_trace: str | None = None

    def save_trace() -> None:
        text = "\n".join(trace_sections) + "\n"
        for output_dir in (WORKDIR, ARTIFACT_DIR):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / trace_path.name).write_text(text)

    def save_submission(raw: str) -> None:
        nonlocal submission_trace
        if submission_adapter is None:
            return
        submission_trace = f"{name}_submission.json"
        for output_dir in (WORKDIR, ARTIFACT_DIR):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / submission_trace).write_text(raw.strip() + "\n")

    def dynamic_search(arguments: dict) -> str:
        query = str(arguments.get("query", "")).strip()
        if len(query) < 8:
            raise ValueError("query must describe the current uncertainty")
        if query in queries:
            raise ValueError("duplicate query; refine it from the latest evidence")
        if len(queries) >= max_queries:
            raise ValueError("dynamic RAG query budget exhausted; submit the artifact")
        result = _search_with_snippets(query, rag_llm, context=context)
        queries.append(query)
        trace_sections.extend([
            f"\n## Query {len(queries)}\n\n{query}",
            f"\n## Result {len(queries)}\n\n{result}",
        ])
        save_trace()
        return result

    def submit(arguments: dict) -> str:
        nonlocal submitted
        if not queries:
            raise ValueError("call search_repo with your own query before submitting")
        raw = (
            submission_adapter(arguments)
            if submission_adapter is not None
            else str(arguments.get("content", ""))
        )
        content = validator(raw)
        output_path.write_text(content)
        save_submission(raw)
        submitted = True
        return f"accepted and wrote {output_path.name}"

    result = run_agent(
        TASK,
        session,
        role_llm,
        max_steps=max_steps,
        compress=False,
        use_tools=True,
        system_prompt=instruction,
        initial_user_message=context,
        action_nudge=(
            f"Call search_repo with a query derived from the current context, "
            f"or call {submit_name} when the artifact is grounded and complete."
        ),
        tool_schemas=[
            SEARCH_REPO_TOOL,
            submit_schema or _submit_tool(submit_name, submit_description),
        ],
        tool_handlers={
            "search_repo": dynamic_search,
            submit_name: submit,
        },
        stop_when=lambda: submitted or len(queries) >= max_queries,
        stop_summary=f"{name} search phase complete",
    )
    synthesis_steps = synthesis_peak = 0
    if queries and not submitted:
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    instruction
                    + "\n\nThe dynamic repository search is complete. You have "
                    "no tools in this synthesis phase. "
                    + (
                        synthesis_instruction
                        or "Return only the required complete artifact; do not "
                        "request or describe more searches."
                    )
                ),
            },
            {
                "role": "user",
                "content": (
                    context
                    + "\n\n# Runtime-generated RAG trace\n\n"
                    + trace_path.read_text(errors="replace")
                ),
            },
        ]
        last_error: str | None = None
        last_candidate: str | None = None
        for _ in range(synthesis_attempts):
            reply = synthesis_llm.chat(synthesis_messages)
            synthesis_steps += 1
            synthesis_peak = max(synthesis_peak, reply.prompt_tokens)
            synthesis_messages.append({"role": "assistant", "content": reply.content})
            candidate = reply.content
            # No-progress guard: a near-identical resubmission cannot clear a
            # validation error, so reject it without re-validating and demand a
            # different, targeted change.
            if (
                last_candidate is not None
                and SequenceMatcher(None, last_candidate, candidate).ratio() > 0.97
            ):
                last_candidate = candidate
                synthesis_messages.append({
                    "role": "user",
                    "content": (
                        "This artifact is essentially identical to your previous "
                        "rejected one, so it fails for the same reason. Change ONLY "
                        "the specific construct the error named (the cited line / AST "
                        "node) and resubmit a materially different artifact."
                    ),
                })
                continue
            last_candidate = candidate
            try:
                output_path.write_text(validator(candidate))
                save_submission(candidate)
                submitted = True
                break
            except Exception as exc:
                message = str(exc)
                repeated = message == last_error
                last_error = message
                correction = f"The synthesized artifact failed validation: {message}. Correct it."
                if repeated:
                    correction += (
                        " This is the SAME error as your previous attempt — your fix "
                        "did not address it. Locate the exact line/construct the error "
                        "names and change that specific code; do not resubmit a similar "
                        "version."
                    )
                synthesis_messages.append({"role": "user", "content": correction})
        _save_messages(f"{name}_synthesis", synthesis_messages)
    _save_role_transcript(name, result)
    if not queries:
        raise RuntimeError(f"{name} submitted no runtime-generated RAG query")
    if not submitted:
        raise RuntimeError(f"{name} failed to synthesize a valid artifact")
    role = {
        "steps": result.steps + synthesis_steps,
        "errors": result.errors,
        "format_errors": result.format_errors,
        "gave_final": submitted,
        "usage": _combined_usage(role_llm, synthesis_llm),
        "peak_ctx_tokens": max(result.peak_ctx_tokens, synthesis_peak),
        "tool_counts": result.tool_counts,
        "command_indexes": [],
        "submission_trace": submission_trace,
    }
    rag = {
        "dynamic": True,
        "trigger": trigger,
        "queries": queries,
        "calls": len(queries),
        "max_queries": max_queries,
        "usage": rag_llm.usage.as_dict(),
        "trace": trace_path.name,
    }
    return role, rag


def _public_contract_passes(session: DockerSession) -> bool:
    return not _public_contract_diagnostics(session)


def _latest_public_evidence(session: DockerSession):
    return extract_structured_evidence(
        session.transcript,
        metric=METRIC,
        expected_num_examples=None,
    )


def _normalization_diagnostics_for_code(
    code: str,
    source: Path,
    *,
    dict_var: str = NORMALIZATION_DICT_VAR,
    key: str = NORMALIZATION_KEY,
) -> list[str]:
    """Compare generated hardcoded normalization against repository source.

    The mechanism is oracle-agnostic; which source variable and dataset key hold
    the reference normalization is passed in (per-oracle config)."""
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
                isinstance(target, ast.Name) and target.id == dict_var
                for target in node.targets
            )
        )
        expected_mean, expected_std = reference_dict[key]
    except (StopIteration, KeyError, SyntaxError, ValueError):
        return []

    # Collect literal assignments anywhere (module level OR inside functions) so
    # a normalization referenced through a local variable is still resolvable;
    # non-literal assignments (e.g. `mean, std = normalization_dict['cifar10']`)
    # are simply left unresolved and skipped — never falsely flagged.
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
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        try:
            mean_node = keywords.get("mean") or node.args[0]
            std_node = keywords.get("std") or node.args[1]
            actual_mean = resolve_literal(mean_node)
            actual_std = resolve_literal(std_node)
        except (IndexError, KeyError, ValueError, TypeError):
            continue
        if list(actual_mean) != list(expected_mean) or list(actual_std) != list(expected_std):
            issue = (
                "Hardcoded CIFAR-10 normalization mismatch with repository source: "
                f"expected mean={expected_mean}, std={expected_std}; "
                f"got mean={actual_mean}, std={actual_std}."
            )
            if issue not in issues:
                issues.append(issue)
    return issues


def _normalization_diagnostics(workdir: Path | None = None) -> list[str]:
    if workdir is None:
        return []
    generated = workdir / "eval_ebo.py"
    if not generated.is_file():
        return []
    return _normalization_diagnostics_for_code(
        generated.read_text(errors="replace"),
        workdir / NORMALIZATION_SOURCE_REL,
    )


def _public_contract_diagnostics(session: DockerSession) -> list[str]:
    """Explain public evidence failures without exposing the private target."""
    evidence = _latest_public_evidence(session)
    if evidence is None:
        malformed = next(
            (
                run for run in reversed(session.transcript)
                if run.ok and "REPRO_RESULT" in run.stdout
            ),
            None,
        )
        if malformed is not None:
            return [
                "A successful evaluation printed REPRO_RESULT, but it was not valid "
                "strict JSON. Serialize the result object with json.dumps.",
                *_normalization_diagnostics(getattr(session, "workdir", None)),
            ]
        issue = "No valid REPRO_RESULT was produced by a successful evaluation command."
        latest = next((run for run in reversed(session.transcript) if not run.ok), None)
        if latest is not None:
            failure = _search_evidence(f"{latest.stdout}\n{latest.stderr}")
            hints = _missing_path_hints(f"{latest.stdout}\n{latest.stderr}")
            if failure:
                issue += f" Fix the latest blocking execution error first:\n{failure}"
            if hints:
                issue += "\nExisting files beside the missing path:\n" + "\n".join(hints)
        return [issue]
    issues: list[str] = []
    if evidence.datasets != EXPECTED_DATASETS:
        issue = f"Dataset counts mismatch: expected {EXPECTED_DATASETS}, got {evidence.datasets}."
        issue += _silent_drop_hint(session, evidence.command_index)
        issues.append(issue)
    if evidence.aggregation != AGGREGATION:
        issues.append(
            f"Aggregation mismatch: expected {AGGREGATION!r}, got {evidence.aggregation!r}."
        )
    if evidence.run_metrics is None:
        issues.append("run_metrics is missing.")
        return issues
    if set(evidence.run_metrics) != set(EXPECTED_RUNS):
        issues.append(
            f"Run names mismatch: expected {EXPECTED_RUNS}, got {sorted(evidence.run_metrics)}."
        )
    expected_names = set(EXPECTED_DATASETS)
    for run, values in evidence.run_metrics.items():
        if set(values) != expected_names:
            issues.append(
                f"Dataset keys for {run} mismatch: expected {sorted(expected_names)}, "
                f"got {sorted(values)}."
            )
    values = [
        value
        for run_values in evidence.run_metrics.values()
        for value in run_values.values()
    ]
    values.append(evidence.actual)
    if not all(0.0 <= value <= 100.0 for value in values) or max(values) <= 1.0:
        issues.append("AUROC values must be finite percentage points in the 0-100 scale.")
    if evidence.run_metrics and all(values for values in evidence.run_metrics.values()):
        computed = sum(
            sum(run_values.values()) / len(run_values)
            for run_values in evidence.run_metrics.values()
        ) / len(evidence.run_metrics)
        if abs(computed - evidence.actual) > 0.011:
            issues.append(
                f"actual does not match dataset_mean_then_run_mean: "
                f"reported {evidence.actual}, recomputed {computed}."
            )
    below_chance = _below_chance_diagnostic(evidence.actual)
    if below_chance:
        issues.append(below_chance)
    issues.extend(_normalization_diagnostics(getattr(session, "workdir", None)))
    return issues


# Generic signals that a data pipeline silently dropped items (PIL/torch/OS —
# not specific to any repo or dataset).
_DROP_SIGNAL_RE = re.compile(
    r"broken|FileNotFoundError|No such file|cannot identify image|truncat|"
    r"could not|UnidentifiedImageError|skipp",
    re.IGNORECASE,
)


def _silent_drop_hint(session, command_index: int | None) -> str:
    """When a count is short, explain it generically: scan the evaluating
    command's own log for signs that items were silently dropped (unreadable /
    missing files), so a short count is diagnosed as 'items were dropped — fix
    the data path/root', not left as a mystery. Oracle-agnostic: the markers are
    standard Python/PIL/OS errors; names no dataset or path."""
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
    """General sanity check for a higher-is-better detection/ranking metric: a
    published method scoring below the random-chance baseline almost always means
    the score or label/decision direction is inverted (such metrics are symmetric
    about chance — an inverted ranking gives ``baseline*2 - value``). This breaks
    the otherwise-blind symmetry between a correct result and its inverse without
    referencing the private target. Oracle-agnostic: ``CHANCE_LEVEL`` is config
    (None disables)."""
    if CHANCE_LEVEL is None or actual >= CHANCE_LEVEL:
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
        # Generic constructs a normalization fix must touch — never the specific
        # (answer-adjacent) constant values, which the diagnostic already names.
        terms.update({"normalize", "std", "mean"})
    if "not valid strict json" in joined:
        terms.update({"json.dumps", "repro_result"})
    missing = re.findall(r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]", joined)
    if missing:
        terms.update(re.findall(r"[a-z0-9_.]+", Path(missing[-1]).name.lower()))
    return terms


def _review_requires_repair(path: Path) -> bool:
    if not path.exists():
        return True
    return "REVIEW_STATUS: PASS" not in path.read_text(errors="replace")


def _round_code_is_endorsed(run_ok: bool, contract_passes: bool, review_path: Path) -> bool:
    """Whether a repair round's new code should be frozen against later edits.

    Endorsement requires ALL THREE signals to agree: a successful execution, the
    deterministic public contract passing, AND the independent Reviewer's PASS.
    Any one disputing keeps the code editable:
      * a bare exit-0 is not enough — a run can print a structurally-valid but
        wrong number (e.g. an inverted EBO/AUROC sign), attempt 026;
      * a Reviewer PASS is not enough either — the Reviewer can miss a mismatch
        the contract still reports (e.g. a short TinyImageNet count), attempt 028.
    Because full agreement also ends the repair loop, this protection is dormant
    during active repair by construction; the anti-regression defense for an
    endorsed fix rests on the patch guards (required_change_terms, minimum
    preservation ratio, unique exact replacement, no-op rejection)."""
    return run_ok and contract_passes and not _review_requires_repair(review_path)


def _repair_loop_should_continue(contract_passes: bool) -> bool:
    """Keep repairing only while the deterministic public contract still FAILS.

    Once it fully passes (counts, runs, aggregation, above-chance sanity,
    composite consistency, normalization) the loop stops — a paranoid Reviewer
    must not push further repairs that can only break a contract-validated result
    (observed in 029/030: the agent reached the correct value, then a later round
    risked flipping it back). The Reviewer still drives repairs while the contract
    is failing, which is its real value; it just loses the power to keep mutating
    an already-validated result."""
    return not contract_passes


def _execute_eval(session: DockerSession):
    syntax = session.shell("python -m py_compile eval_ebo.py", timeout=120)
    if not syntax.ok:
        return syntax
    return session.shell(f"python eval_ebo.py --root {CHECKPOINT_ROOT}")


def main() -> None:
    _copy_clean_source()
    _assert_blind_workspace()
    shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)

    session = DockerSession(
        WORKDIR, image=IMAGE, mem="6g", cpus=6.0, default_timeout=1800,
    )
    session.go_offline()
    roles: dict[str, dict] = {}
    rag: dict[str, dict] = {}
    protected_code_blocks: set[str] = set()
    workflow_error: str | None = None
    try:
        roles["navigator"], rag["navigator"] = _dynamic_rag_role(
            name="navigator",
            session=session,
            instruction=f"""You are the Navigator in a collaborative ML
reproduction team. You receive no prewritten repository queries. Identify the
most important unknowns in the task, formulate your own search_repo query, use
the retrieved source to refine later queries when needed, then submit a concise
grounded handoff. Include exact source paths, EBO and AUROC semantics, data and
preprocessing, checkpoint layout, aggregation, and CPU/dependency risks.
Do not guess or mention the private target.

Task:
{TASK}""",
            context=TASK,
            output_path=WORKDIR / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=_validate_report,
            trigger="initial_task",
            max_steps=7,
        )

        builder_context = (
            "# Navigator handoff\n\n"
            + (WORKDIR / "navigator_report.md").read_text(errors="replace")
        )
        roles["reproducer"], rag["reproducer"] = _dynamic_rag_role(
            name="reproducer",
            session=session,
            instruction=f"""You are the Reproducer/Builder. Generate a complete
CPU-safe `eval_ebo.py`. You receive a Navigator handoff but no prewritten RAG
queries. Before coding, identify an implementation uncertainty and call
search_repo with your own query. Use follow-up searches only when retrieved
source exposes another uncertainty, then submit the complete script.

Public execution contract:
- exact OpenOOD ResNet18_32x32 and official s0/s1/s2 checkpoints;
- official CIFAR-10 preprocessing and benchmark image lists; do not reimplement
  the repository's ImglistDataset;
- implement the small torchvision test transform directly from
  `openood/preprocessors/transform.py`; do not parse checkpoint `config.yml`
  files or instantiate `TestStandardPreProcessor`;
- EBO and official AUROC sign semantics, percentage points, and dataset-then-run mean;
- print exactly one strict-JSON `REPRO_RESULT` using `json.dumps`; its
  `datasets` values are evaluated sample counts, not checkpoint/run counts;
- accept `--root` and use batched DataLoader CPU inference;
- import the model and dataset only from direct modules such as
  `openood.networks.resnet18_32x32` and
  `openood.datasets.imglist_dataset`; do not import `openood.evaluation_api`,
  `openood.evaluators`, or `openood.postprocessors`, because their package
  initializers pull unrelated optional dependencies;
- implement the small EBO score and AUROC calculation locally from retrieved
  repository semantics;
- {EVIDENCE}

Do not guess or mention the private target.""",
            context=builder_context,
            output_path=WORKDIR / "eval_ebo.py",
            submit_name="submit_code",
            submit_description="Submit the complete generated eval_ebo.py.",
            validator=_validate_code,
            trigger="navigator_handoff",
            max_steps=7,
            synthesis_attempts=5,
        )
        critic_context = (
            "# Generated evaluation script\n\n"
            + (WORKDIR / "eval_ebo.py").read_text(errors="replace")
            + "\n\n# Navigator handoff\n\n"
            + (WORKDIR / "navigator_report.md").read_text(errors="replace")
        )
        roles["critic"], rag["critic"] = _dynamic_rag_role(
            name="critic",
            session=session,
            instruction=f"""You are an independent Code Critic. Audit the
generated evaluation script against repository source. You receive no
prewritten queries: choose a search_repo query targeting the highest-risk
unverified claim in the code, and issue follow-up queries only when evidence
requires them. Submit a complete corrected script, not a prose review.

Verify model import, benchmark paths, preprocessing, EBO/AUROC sign,
percentage units, s0/s1/s2 aggregation, `--root`, and batched CPU execution.
Require strict-JSON `REPRO_RESULT` via `json.dumps`, with evaluated sample
counts in `datasets`, not checkpoint/run counts.
Treat every hardcoded normalization value and any custom Dataset
implementation as high risk: verify them against repository source and prefer
the official ImglistDataset.
Use a small direct torchvision test transform from repository normalization
source. Reject checkpoint `config.yml` parsing and `TestStandardPreProcessor`,
which add irrelevant serialized-config failure modes.
Allow only direct OpenOOD model/dataset module imports. Reject
`openood.evaluation_api`, `openood.evaluators`, and `openood.postprocessors`;
use a minimal local EBO/AUROC implementation instead.
The script must satisfy:
{EVIDENCE}

Do not guess or mention the private target.""",
            context=critic_context,
            output_path=WORKDIR / "eval_ebo.py",
            submit_name="submit_code",
            submit_description="Submit the complete audited and corrected eval_ebo.py.",
            validator=_validate_code,
            trigger="generated_code_audit",
            max_steps=7,
            synthesis_attempts=5,
        )
        execution_start = len(session.transcript)
        eval_run = _execute_eval(session)
        roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        roles["reproducer"]["command_indexes"] = [execution_start + 1, len(session.transcript)]
        session.write_file(
            "reproducer_public_log.txt",
            _public_log(session, execution_start),
        )

        def review_current(round_index: int) -> None:
            diagnostics = _public_contract_diagnostics(session)
            review_context = (
                "# Navigator handoff\n\n"
                + (WORKDIR / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Evaluation implementation\n\n"
                + _clip((WORKDIR / "eval_ebo.py").read_text(errors="replace"), 12000)
                + "\n\n# Public execution logs\n\n"
                + _clip(
                    (WORKDIR / "reproducer_public_log.txt").read_text(errors="replace"),
                    12000,
                )
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"reviewer_{round_index}"
            roles[key], rag[key] = _dynamic_rag_role(
                name=key,
                session=session,
                instruction="""You are the independent Reviewer. Audit the
current implementation and public execution log. You receive no prewritten
queries. Derive a search_repo query from the concrete execution error or the
highest-risk semantic claim in the current code. Use repository evidence to
explain the finding. The deterministic public-contract audit is authoritative:
do not ignore its failures, and do not request changes to behavior already
demonstrated by a successful execution unless repository evidence proves a
semantic mismatch. When execution failed, focus the review on the latest
blocking error; defer unrelated semantic concerns until the program runs.
End with exactly `REVIEW_STATUS: PASS` only when no repair is needed; otherwise
end with exactly `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target.""",
                context=review_context,
                output_path=WORKDIR / "review_report.md",
                submit_name="submit_review",
                submit_description="Submit the source-grounded execution audit.",
                validator=_validate_review,
                trigger="execution_result" if round_index == 0 else "repair_execution_result",
                max_steps=6,
                max_queries=2,
            )

        review_current(0)
        for round_index in (1, 2, 3, 4):
            if not _repair_loop_should_continue(_public_contract_passes(session)):
                break
            diagnostics = _public_contract_diagnostics(session)
            repair_context = (
                "# Current evaluation script\n\n"
                + (WORKDIR / "eval_ebo.py").read_text(errors="replace")
                + "\n\n# Public execution log\n\n"
                + _public_log(session, execution_start)
                + "\n\n# Independent reviewer audit\n\n"
                + (WORKDIR / "review_report.md").read_text(errors="replace")
                + "\n\n# Navigator handoff\n\n"
                + (WORKDIR / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"repair_{round_index}"
            accepted_new_blocks: list[str] = []
            roles[key], rag[key] = _dynamic_rag_role(
                name=key,
                session=session,
                instruction=f"""You are Repair Agent {round_index}. Fix the
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

Preserve working behavior and the public contract: percentage AUROC, correct
EBO/AUROC direction, exact dataset counts, s0/s1/s2 dataset-then-run mean,
strict-JSON `REPRO_RESULT` via `json.dumps`, `--root`, batched DataLoader CPU
inference, and no unrelated broad-package imports.
{EVIDENCE}

Do not guess or mention the private target.""",
                context=repair_context,
                output_path=WORKDIR / "eval_ebo.py",
                submit_name="submit_patch",
                submit_description=(
                    "Submit small exact-replacement edits for the current eval_ebo.py."
                ),
                validator=lambda payload: _apply_code_patch(
                    WORKDIR / "eval_ebo.py",
                    payload,
                    protected_blocks=protected_code_blocks,
                    required_change_terms=_diagnostic_change_terms(diagnostics),
                    accepted_new_blocks=accepted_new_blocks,
                ),
                trigger="execution_error_and_reviewer_finding",
                max_steps=7,
                max_queries=2,
                submit_schema=_patch_tool(
                    "submit_patch",
                    "Submit small exact-replacement edits for the current eval_ebo.py.",
                    max_items=2,
                ),
                submission_adapter=lambda arguments: json.dumps(arguments),
                synthesis_instruction=(
                    "Return only one JSON object with `edits` and `rationale`, "
                    "using the submit_patch schema. Every `old` string must come "
                    "from the current eval_ebo.py. Do not patch repository source "
                    "or return the complete file."
                ),
                synthesis_attempts=4,
            )
            start = len(session.transcript)
            repaired_run = _execute_eval(session)
            roles[f"repair_{round_index}"]["errors"] = 0 if repaired_run.ok else 1
            roles[f"repair_{round_index}"]["command_indexes"] = [
                start + 1,
                len(session.transcript),
            ]
            session.write_file(
                "reproducer_public_log.txt",
                _public_log(session, execution_start),
            )
            review_current(round_index)
            # Freeze this round's new code only once the INDEPENDENT REVIEWER
            # endorses the result (REVIEW_STATUS: PASS), not on a bare exit-0: a
            # successful run can still print a structurally-valid but semantically
            # wrong number (e.g. an inverted EBO/AUROC sign the blind public
            # contract cannot detect). Reviewer-endorsed code is protected from
            # silent regression; code the reviewer still disputes stays editable
            # so a later Repair can revisit it.
            if _round_code_is_endorsed(
                repaired_run.ok,
                _public_contract_passes(session),
                WORKDIR / "review_report.md",
            ):
                protected_code_blocks.update(accepted_new_blocks)
    except Exception as exc:
        workflow_error = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()

    verdict = verify_run(
        session.transcript,
        session.workdir,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        metric=METRIC,
        expected_num_examples=None,
        expected_datasets=EXPECTED_DATASETS,
        expected_runs=EXPECTED_RUNS,
        expected_aggregation=AGGREGATION,
    )
    public_evidence = extract_structured_evidence(
        session.transcript,
        metric=METRIC,
        expected_num_examples=None,
        expected_datasets=EXPECTED_DATASETS,
        expected_runs=EXPECTED_RUNS,
        expected_aggregation=AGGREGATION,
    )
    rag_requirement = bool(rag) and all(
        stage["dynamic"] and stage["calls"] >= 1 for stage in rag.values()
    )
    handoff_requirement = (
        (WORKDIR / "navigator_report.md").exists()
        and (WORKDIR / "review_report.md").exists()
    )
    collaboration_pass = verdict.match and rag_requirement and handoff_requirement
    total_cost = round(
        sum(r["usage"].get("cost_yuan", 0.0) for r in roles.values())
        + sum(stage["usage"].get("cost_yuan", 0.0) for stage in rag.values()),
        4,
    )
    output = {
        "task": TASK,
        "blind_workspace_checked": True,
        "agents": len(roles),
        "attempt": ATTEMPT,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "retrieval_ranker": "exact_path_symbol_plus_bm25_llm",
        "repair_mode": "structured_exact_replacement_patch",
        "workflow_error": workflow_error,
        "total_rag_calls": sum(stage["calls"] for stage in rag.values()),
        "rag_requirement_met": rag_requirement,
        "handoff_requirement_met": handoff_requirement,
        "public_evidence_found": public_evidence is not None,
        "public_contract_diagnostics": _public_contract_diagnostics(session),
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": len(session.transcript),
    }
    result_json = json.dumps(output, indent=2) + "\n"
    replay_script = session.replay_script() + "\n"
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(result_json)
        (output_dir / "commands.sh").write_text(replay_script)
        for handoff in (
            "navigator_report.md",
            "review_report.md",
            "reproducer_public_log.txt",
            "eval_ebo.py",
        ):
            source = WORKDIR / handoff
            if source.exists() and output_dir != WORKDIR:
                shutil.copy2(source, output_dir / handoff)
        if output_dir != WORKDIR:
            for trace in WORKDIR.glob("*_rag_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)
    print(result_json)


if __name__ == "__main__":
    main()
