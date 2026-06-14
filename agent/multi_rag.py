"""Generic Multi-Agent + Dynamic RAG orchestration skeleton.

Shared helpers, _dynamic_rag_role, OracleConfig, and run_oracle.
Oracle-specific config (prompts, validators, contract checks) lives in
evals/oracles/<oracle>.py; this module contains zero task literals.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from agent.llm import ChatLLM
from agent.loop import AgentResult, TOOLS, run_agent
from retrieval.search import relevant_snippet, search_repo
from verify.check import verify_run

SEARCH_REPO_TOOL = next(t for t in TOOLS if t["function"]["name"] == "search_repo")


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------

def _clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


def _public_log(session: Any, start: int) -> str:
    parts = []
    for index, run in enumerate(session.transcript[start:], start + 1):
        parts.append(
            f"## Command {index}\n\n```bash\n{run.command}\n```\n\n"
            f"exit={run.exit_code} timed_out={run.timed_out}\n\n"
            f"```text\n{_clip(run.stdout)}\n{_clip(run.stderr)}\n```\n"
        )
    return "\n".join(parts)


def _save_role_transcript(
    name: str, result: AgentResult, workdir: Path, artifact_dir: Path
) -> None:
    text = "".join(json.dumps(m) + "\n" for m in result.transcript)
    for d in (workdir, artifact_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_transcript.jsonl").write_text(text)


def _save_messages(
    name: str, messages: list[dict], workdir: Path, artifact_dir: Path
) -> None:
    text = "".join(json.dumps(m) + "\n" for m in messages)
    for d in (workdir, artifact_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_transcript.jsonl").write_text(text)


def _combined_usage(*llms: ChatLLM) -> dict:
    usages = [llm.usage.as_dict() for llm in llms]
    return {
        "llm_calls": sum(u["llm_calls"] for u in usages),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
        "cache_hit_tokens": sum(u["cache_hit_tokens"] for u in usages),
        "completion_tokens": sum(u["completion_tokens"] for u in usages),
        "cost_yuan": round(sum(u["cost_yuan"] for u in usages), 4),
    }


def _extract_python(text: str) -> str:
    """Select the eval script from a model reply — prefers REPRO_RESULT-bearing block."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [b for b in blocks if "REPRO_RESULT" in b] or blocks
    return max(candidates, key=len).strip() + "\n"


# ---------------------------------------------------------------------------
# Tool schema builders
# ---------------------------------------------------------------------------

def _submit_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Complete artifact content."},
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
                                "old": {"type": "string", "description": "Exact existing code occurring once."},
                                "new": {"type": "string", "description": "Replacement code."},
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


# ---------------------------------------------------------------------------
# Shared report / review validators
# ---------------------------------------------------------------------------

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


def _review_requires_repair(path: Path) -> bool:
    if not path.exists():
        return True
    return "REVIEW_STATUS: PASS" not in path.read_text(errors="replace")


# ---------------------------------------------------------------------------
# Search utilities
# ---------------------------------------------------------------------------

def _search_evidence(context: str) -> str:
    traceback_paths = re.findall(r'File "/workspace/([^"]+\.py)"', context)
    mentioned_paths = re.findall(
        r"\b([A-Za-z0-9_][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]+\.py)\b", context
    )
    failures = re.findall(
        r"(?m)^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|"
        r"ModuleNotFoundError|RuntimeError|TypeError|ValueError):.*$",
        context,
    )
    paths = traceback_paths[-4:] or mentioned_paths[-3:]
    return "\n".join(dict.fromkeys(paths + failures[-1:]))[-2400:]


def _missing_path_hints(context: str, workdir: Path) -> list[str]:
    matches = re.findall(r"FileNotFoundError:.*?['\"]([^'\"]+)['\"]", context)
    if not matches:
        return []
    missing = matches[-1]
    relative = missing.removeprefix("/workspace/").lstrip("./")
    parent = (workdir / relative).parent
    if not parent.is_dir():
        ancestor = parent
        while not ancestor.is_dir() and workdir in ancestor.parents:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            return []
        try:
            rel = ancestor.relative_to(workdir)
        except ValueError:
            return []
        prefix = "" if str(rel) == "." else f"{rel}/"
        return sorted(
            f"{prefix}{c.name}" + ("/" if c.is_dir() else "")
            for c in ancestor.iterdir()
        )[:8]
    stem_tokens = set(re.findall(r"[a-z0-9]+", Path(relative).stem.lower()))
    candidates = [
        (len(stem_tokens & set(re.findall(r"[a-z0-9]+", p.stem.lower()))), p.name)
        for p in parent.iterdir()
        if p.is_file()
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    prefix = str(Path(relative).parent)
    return [f"{prefix}/{name}" for _, name in candidates[:8]]


def _search_with_snippets(
    query: str,
    llm: ChatLLM,
    workdir: Path,
    *,
    context: str | None = None,
    extra_exclude: set[str] | None = None,
    max_files: int = 4,
) -> str:
    generated = set(extra_exclude or ())
    generated.update(p.name for p in workdir.glob("*_rag_trace.md"))
    generated.update(p.name for p in workdir.glob("*_transcript.jsonl"))
    ranking_evidence = _search_evidence(context or "")
    path_hints = _missing_path_hints(context or "", workdir)
    if path_hints:
        ranking_evidence += "\nExisting files beside the missing path:\n" + "\n".join(path_hints)
    result = search_repo(
        query,
        workdir,
        llm,
        exclude_paths=generated,
        context=ranking_evidence or None,
    )
    paths: list[str] = []
    for line in result.splitlines():
        m = re.match(r"^\s{2}(\S+)\s+—", line)
        if m and m.group(1) not in paths:
            paths.append(m.group(1))
    snippets = []
    for rel in paths[:max_files]:
        p = workdir / rel
        if p.is_file():
            snippets.append(
                f"\n## Source: {rel}\n\n"
                f"{relevant_snippet(p, f'{query}\n{ranking_evidence}', 3200)}"
            )
    evidence_section = (
        f"\n\nError evidence used for ranking:\n{ranking_evidence}" if ranking_evidence else ""
    )
    return (
        result
        + evidence_section
        + "\n\nRetrieved source snippets:\n"
        + "\n".join(snippets)
    )


# ---------------------------------------------------------------------------
# Patch utilities (used by oracle configs that choose patch-based repair)
# ---------------------------------------------------------------------------

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
    validate_code: Callable[[str], str],
    protected_blocks: set[str] | None = None,
    required_change_terms: set[str] | None = None,
    accepted_new_blocks: list[str] | None = None,
) -> str:
    """Apply a JSON-structured exact-replacement patch and validate the result."""
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
            hint = f"\nClosest actual code currently in the file:\n{nearby}" if nearby else ""
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
        for tag, o0, o1, n0, n1 in SequenceMatcher(None, old, new).get_opcodes():
            if tag != "equal":
                changed_fragments.extend((old[o0:o1], new[n0:n1]))
        updated = updated.replace(old, new, 1)
    if changed_chars > len(current) * 0.65:
        raise ValueError("patch replaces too much of the current file")
    if SequenceMatcher(None, current, updated).ratio() < 0.55:
        raise ValueError("patch does not preserve enough working code")
    for block in protected_blocks or set():
        if block in current and block not in updated:
            raise ValueError(
                "patch changes code already confirmed by a reviewer-endorsed execution"
            )
    if required_change_terms:
        changed = "\n".join(changed_fragments).lower()
        if not any(term.lower() in changed for term in required_change_terms):
            raise ValueError(
                f"patch does not address deterministic public-contract issue: "
                f"{sorted(required_change_terms)}"
            )
    if accepted_new_blocks is not None:
        accepted_new_blocks.extend(
            edit["new"]
            for edit in edits
            if isinstance(edit.get("new"), str) and len(edit["new"]) >= 12
        )
    return validate_code(updated)


# ---------------------------------------------------------------------------
# Dynamic RAG role
# ---------------------------------------------------------------------------

def _dynamic_rag_role(
    *,
    name: str,
    task: str,
    workdir: Path,
    artifact_dir: Path,
    session: Any,
    instruction: str,
    context: str,
    output_path: Path,
    submit_name: str,
    submit_description: str,
    validator: Callable[[str], str],
    trigger: str,
    search_extra_exclude: set[str] | None = None,
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
    trace_path = workdir / f"{name}_rag_trace.md"
    submission_trace: str | None = None

    def save_trace() -> None:
        text = "\n".join(trace_sections) + "\n"
        for d in (workdir, artifact_dir):
            d.mkdir(parents=True, exist_ok=True)
            (d / trace_path.name).write_text(text)

    def save_submission(raw: str) -> None:
        nonlocal submission_trace
        if submission_adapter is None:
            return
        submission_trace = f"{name}_submission.json"
        for d in (workdir, artifact_dir):
            d.mkdir(parents=True, exist_ok=True)
            (d / submission_trace).write_text(raw.strip() + "\n")

    def dynamic_search(arguments: dict) -> str:
        query = str(arguments.get("query", "")).strip()
        if len(query) < 8:
            raise ValueError("query must describe the current uncertainty")
        if query in queries:
            raise ValueError("duplicate query; refine it from the latest evidence")
        if len(queries) >= max_queries:
            raise ValueError("dynamic RAG query budget exhausted; submit the artifact")
        result = _search_with_snippets(
            query, rag_llm, workdir,
            context=context,
            extra_exclude=search_extra_exclude,
        )
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
        task,
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
            try:
                validated = validator(candidate)
            except Exception as exc:
                message = str(exc)
                near_identical = (
                    last_candidate is not None
                    and SequenceMatcher(None, last_candidate, candidate).ratio() > 0.97
                )
                repeated = message == last_error
                last_candidate, last_error = candidate, message
                correction = (
                    f"The synthesized artifact failed validation: {message}. Correct it."
                )
                if near_identical:
                    correction += (
                        " Your artifact barely changed and still fails — change the "
                        "EXACT construct the error names (the cited line / AST node)."
                    )
                elif repeated:
                    correction += (
                        " This is the SAME error as your previous attempt — locate the "
                        "exact construct the error names and change only that."
                    )
                synthesis_messages.append({"role": "user", "content": correction})
                continue
            output_path.write_text(validated)
            save_submission(candidate)
            submitted = True
            break
        _save_messages(f"{name}_synthesis", synthesis_messages, workdir, artifact_dir)

    _save_role_transcript(name, result, workdir, artifact_dir)
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


# ---------------------------------------------------------------------------
# Oracle configuration
# ---------------------------------------------------------------------------

@dataclass
class OracleConfig:
    # Identity
    name: str
    task: str
    metric: str
    expected: float
    tolerance: float
    attempt: str

    # Paths
    workdir: Path
    artifact_dir: Path
    eval_script: str  # file name only, e.g. "eval_ebo.py"

    # Session lifecycle
    make_session: Callable[[], Any]
    session_go_offline: bool = False

    # Oracle lifecycle
    copy_clean_source: Callable[[], None] = field(default=lambda: None)
    execute_eval: Callable[[Any], Any] = field(default=lambda s: None)

    # Validation
    validate_code: Callable[[str], str] = field(default=lambda s: s)
    validate_report: Callable[[str], str] = field(default_factory=lambda: _validate_report)
    validate_review: Callable[[str], str] = field(default_factory=lambda: _validate_review)

    # Contract
    public_contract_passes: Callable[[Any], bool] = field(default=lambda s: True)
    public_contract_diagnostics: Callable[[Any], list[str]] = field(default=lambda s: [])

    # Verify kwargs (forwarded to verify_run)
    verify_kwargs: dict = field(default_factory=dict)

    # Role instructions
    navigator_instruction: str = ""
    reproducer_instruction: str = ""
    critic_instruction: str = ""
    reviewer_instruction: str = ""
    repair_instruction: str = ""  # may contain {round_index}

    # Repair configuration
    repair_mode_label: str = "full_file_replacement"
    repair_submit_name: str = "submit_code"
    repair_submit_description: str = "Submit the repaired script."
    repair_submit_schema: dict | None = None
    repair_submission_adapter: Callable[[dict], str] | None = None
    repair_synthesis_instruction: str | None = None
    # Factory: (diagnostics, protected_blocks, accepted_new_blocks) -> validator
    # If None, falls back to validate_code for repair rounds.
    repair_make_validator: (
        Callable[[list[str], set[str], list[str]], Callable[[str], str]] | None
    ) = None
    # (run_ok, contract_passes, review_path) -> should we freeze accepted_new_blocks?
    make_endorsed: Callable[[bool, bool, Path], bool] | None = None

    # File names excluded from search (oracle-generated files, e.g. "eval_ebo.py")
    search_extra_exclude: set[str] = field(default_factory=set)

    # Blind workspace check (optional)
    assert_blind_workspace: Callable[[], None] | None = None

    # Files copied to artifact_dir after the run
    handoff_files: tuple[str, ...] = (
        "navigator_report.md",
        "review_report.md",
        "reproducer_public_log.txt",
    )

    retrieval_ranker: str = "exact_path_symbol_plus_bm25_llm"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_oracle(config: OracleConfig) -> None:
    config.copy_clean_source()
    if config.assert_blind_workspace is not None:
        config.assert_blind_workspace()
    shutil.rmtree(config.artifact_dir, ignore_errors=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    session = config.make_session()
    if config.session_go_offline:
        session.go_offline()

    def rag_role(**kwargs: Any) -> tuple[dict, dict]:
        return _dynamic_rag_role(
            task=config.task,
            workdir=config.workdir,
            artifact_dir=config.artifact_dir,
            session=session,
            search_extra_exclude=config.search_extra_exclude,
            **kwargs,
        )

    roles: dict[str, dict] = {}
    rag: dict[str, dict] = {}
    protected_code_blocks: set[str] = set()
    workflow_error: str | None = None
    execution_start = 0

    try:
        roles["navigator"], rag["navigator"] = rag_role(
            name="navigator",
            instruction=config.navigator_instruction,
            context=config.task,
            output_path=config.workdir / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=config.validate_report,
            trigger="initial_task",
            max_steps=7,
        )

        builder_context = (
            "# Navigator handoff\n\n"
            + (config.workdir / "navigator_report.md").read_text(errors="replace")
        )
        roles["reproducer"], rag["reproducer"] = rag_role(
            name="reproducer",
            instruction=config.reproducer_instruction,
            context=builder_context,
            output_path=config.workdir / config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete generated {config.eval_script}.",
            validator=config.validate_code,
            trigger="navigator_handoff",
            max_steps=7,
            synthesis_attempts=5,
        )

        critic_context = (
            "# Generated evaluation script\n\n"
            + (config.workdir / config.eval_script).read_text(errors="replace")
            + "\n\n# Navigator handoff\n\n"
            + (config.workdir / "navigator_report.md").read_text(errors="replace")
        )
        roles["critic"], rag["critic"] = rag_role(
            name="critic",
            instruction=config.critic_instruction,
            context=critic_context,
            output_path=config.workdir / config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete audited {config.eval_script}.",
            validator=config.validate_code,
            trigger="generated_code_audit",
            max_steps=7,
            synthesis_attempts=5,
        )

        execution_start = len(session.transcript)
        eval_run = config.execute_eval(session)
        roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        roles["reproducer"]["command_indexes"] = [
            execution_start + 1, len(session.transcript)
        ]
        session.write_file(
            "reproducer_public_log.txt",
            _public_log(session, execution_start),
        )

        def review_current(round_index: int) -> None:
            diagnostics = config.public_contract_diagnostics(session)
            review_context = (
                "# Navigator handoff\n\n"
                + (config.workdir / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Evaluation implementation\n\n"
                + _clip(
                    (config.workdir / config.eval_script).read_text(errors="replace"),
                    12000,
                )
                + "\n\n# Public execution logs\n\n"
                + _clip(
                    (config.workdir / "reproducer_public_log.txt").read_text(
                        errors="replace"
                    ),
                    12000,
                )
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"reviewer_{round_index}"
            roles[key], rag[key] = rag_role(
                name=key,
                instruction=config.reviewer_instruction,
                context=review_context,
                output_path=config.workdir / "review_report.md",
                submit_name="submit_review",
                submit_description="Submit the source-grounded execution audit.",
                validator=config.validate_review,
                trigger=(
                    "execution_result" if round_index == 0
                    else "repair_execution_result"
                ),
                max_steps=6,
                max_queries=2,
            )

        review_current(0)

        for round_index in (1, 2, 3, 4):
            if config.public_contract_passes(session):
                break
            diagnostics = config.public_contract_diagnostics(session)
            repair_context = (
                "# Current evaluation script\n\n"
                + (config.workdir / config.eval_script).read_text(errors="replace")
                + "\n\n# Public execution log\n\n"
                + _public_log(session, execution_start)
                + "\n\n# Independent reviewer audit\n\n"
                + (config.workdir / "review_report.md").read_text(errors="replace")
                + "\n\n# Navigator handoff\n\n"
                + (config.workdir / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"repair_{round_index}"
            accepted_new_blocks: list[str] = []
            if config.repair_make_validator is not None:
                repair_validator = config.repair_make_validator(
                    diagnostics, protected_code_blocks, accepted_new_blocks
                )
            else:
                repair_validator = config.validate_code

            roles[key], rag[key] = rag_role(
                name=key,
                instruction=config.repair_instruction.format(round_index=round_index),
                context=repair_context,
                output_path=config.workdir / config.eval_script,
                submit_name=config.repair_submit_name,
                submit_description=config.repair_submit_description,
                validator=repair_validator,
                trigger="execution_error_and_reviewer_finding",
                max_steps=7,
                max_queries=2,
                submit_schema=config.repair_submit_schema,
                submission_adapter=config.repair_submission_adapter,
                synthesis_instruction=config.repair_synthesis_instruction,
                synthesis_attempts=4,
            )

            start = len(session.transcript)
            repaired_run = config.execute_eval(session)
            roles[key]["errors"] = 0 if repaired_run.ok else 1
            roles[key]["command_indexes"] = [start + 1, len(session.transcript)]
            session.write_file(
                "reproducer_public_log.txt",
                _public_log(session, execution_start),
            )
            review_current(round_index)

            if config.make_endorsed is not None and config.make_endorsed(
                repaired_run.ok,
                config.public_contract_passes(session),
                config.workdir / "review_report.md",
            ):
                protected_code_blocks.update(accepted_new_blocks)

    except Exception as exc:
        workflow_error = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()

    verdict = verify_run(
        session.transcript,
        config.workdir,
        expected=config.expected,
        tolerance=config.tolerance,
        metric=config.metric,
        **config.verify_kwargs,
    )

    rag_requirement = bool(rag) and all(
        stage["dynamic"] and stage["calls"] >= 1 for stage in rag.values()
    )
    handoff_requirement = (
        (config.workdir / "navigator_report.md").exists()
        and (config.workdir / "review_report.md").exists()
    )
    collaboration_pass = verdict.match and rag_requirement and handoff_requirement
    total_cost = round(
        sum(r["usage"].get("cost_yuan", 0.0) for r in roles.values())
        + sum(s["usage"].get("cost_yuan", 0.0) for s in rag.values()),
        4,
    )
    output = {
        "task": config.task,
        "blind_workspace_checked": config.assert_blind_workspace is not None,
        "agents": len(roles),
        "attempt": config.attempt,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "retrieval_ranker": config.retrieval_ranker,
        "repair_mode": config.repair_mode_label,
        "workflow_error": workflow_error,
        "total_rag_calls": sum(stage["calls"] for stage in rag.values()),
        "rag_requirement_met": rag_requirement,
        "handoff_requirement_met": handoff_requirement,
        "public_evidence_found": config.public_contract_passes(session),
        "public_contract_diagnostics": config.public_contract_diagnostics(session),
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": len(session.transcript),
    }
    result_json = json.dumps(output, indent=2) + "\n"

    replay_fn = getattr(session, "replay_script", None)
    replay_script = (replay_fn() + "\n") if replay_fn is not None else None

    for output_dir in (config.workdir, config.artifact_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(result_json)
        if replay_script is not None:
            (output_dir / "commands.sh").write_text(replay_script)
        for handoff in config.handoff_files:
            src = config.workdir / handoff
            if src.exists() and output_dir != config.workdir:
                shutil.copy2(src, output_dir / handoff)
        src_eval = config.workdir / config.eval_script
        if src_eval.exists() and output_dir != config.workdir:
            shutil.copy2(src_eval, output_dir / config.eval_script)
        if output_dir != config.workdir:
            for trace in config.workdir.glob("*_rag_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)

    print(result_json)
