"""Generic Multi-Agent + Dynamic RAG orchestration skeleton.

Shared helpers, _dynamic_rag_role, OracleConfig, and run_oracle.
Oracle-specific config (prompts, validators, contract checks) lives in
evals/oracles/<oracle>.py; this module contains zero task literals.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from agent.generic_prompts import GENERIC_PROMPTS, RolePrompts
from agent.llm import ChatLLM
from agent.loop import AgentResult, TOOLS, run_agent
from retrieval.search import relevant_snippet, search_repo
from verify.check import verify_run

SEARCH_REPO_TOOL = next(t for t in TOOLS if t["function"]["name"] == "search_repo")
MAX_RUNTIME_PROBES = 8
MAX_RUNTIME_PROBES_PER_ROLE = 2
RUNTIME_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "runtime_probe",
        "description": (
            "Run one restricted diagnostic in the provisioned runtime. "
            "Use it to check an import, inspect a Python signature, list a local "
            "path, or request CLI help. It cannot run arbitrary shell commands or "
            "the full evaluation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "import_smoke",
                        "python_signature",
                        "path_list",
                        "cli_help",
                    ],
                },
                "target": {
                    "type": "string",
                    "description": (
                        "A dotted Python module/object, or a workspace-relative "
                        "path for path_list/cli_help."
                    ),
                },
            },
            "required": ["kind", "target"],
        },
    },
}


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------

def _clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def _runtime_probe_command(kind: str, target: str) -> str:
    """Build a command from a small diagnostic vocabulary, never raw shell."""
    kind = kind.strip()
    target = target.strip()
    if len(target) > 240:
        raise ValueError("runtime probe target is too long")

    if kind in {"import_smoke", "python_signature"}:
        if not _DOTTED_NAME.fullmatch(target):
            raise ValueError("Python probe target must be a dotted identifier")
        if kind == "import_smoke":
            code = (
                "import importlib; "
                f"m=importlib.import_module({target!r}); "
                "print('IMPORTED', m.__name__); "
                "print('FILE', getattr(m, '__file__', None)); "
                "print('VERSION', getattr(m, '__version__', None))"
            )
        else:
            code = f"""import importlib
import inspect
parts = {target!r}.split(".")
obj = None
attrs = []
for index in range(len(parts), 0, -1):
    module_name = ".".join(parts[:index])
    try:
        obj = importlib.import_module(module_name)
        attrs = parts[index:]
        break
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(exc.name + "."):
            raise
if obj is None:
    raise ModuleNotFoundError({target!r})
for attr in attrs:
    obj = getattr(obj, attr)
print("OBJECT", {target!r})
print("SIGNATURE", inspect.signature(obj))
print("SOURCE", inspect.getsourcefile(obj))
"""
        return f"python -c {shlex.quote(code)}"

    path = Path(target)
    if not target or path.is_absolute() or ".." in path.parts:
        raise ValueError("path probe target must stay inside the workspace")
    if kind == "path_list":
        code = f"""from pathlib import Path
root = Path.cwd()
path = root / {target!r}
print("PATH", path.relative_to(root), "RESOLVED", path.resolve())
print("EXISTS", path.exists(), "FILE", path.is_file(), "DIR", path.is_dir())
if path.is_file():
    print(path.relative_to(root))
elif path.is_dir():
    for index, item in enumerate(sorted(p for p in path.rglob("*") if p.is_file())):
        if index >= 100:
            print("... output capped at 100 files")
            break
        print(item.relative_to(root))
"""
        return f"python -c {shlex.quote(code)}"
    if kind == "cli_help":
        if path.suffix != ".py":
            raise ValueError("cli_help target must be a workspace-relative .py file")
        return f"python {shlex.quote(target)} --help"
    raise ValueError(f"unsupported runtime probe kind: {kind!r}")


def _runtime_probe_observation(run: Any) -> str:
    status = (
        f"timed out after {run.duration_s:.0f}s"
        if run.timed_out
        else f"exit {run.exit_code} in {run.duration_s:.0f}s"
    )
    return (
        f"Restricted runtime probe ({status}).\n"
        f"stdout:\n{_clip(run.stdout, 4000)}\n"
        f"stderr:\n{_clip(run.stderr, 4000)}"
    )


def _public_log(session: Any, start: int) -> str:
    parts = []
    for index, run in enumerate(session.transcript[start:], start + 1):
        parts.append(
            f"## Command {index}\n\n```bash\n{run.command}\n```\n\n"
            f"exit={run.exit_code} timed_out={run.timed_out}\n\n"
            f"```text\n{_clip(run.stdout)}\n{_clip(run.stderr)}\n```\n"
        )
    return "\n".join(parts)


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish generated code atomically for Docker bind-mount readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


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
    """Select the eval script from a model reply, preferring result-producing code."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [
        b for b in blocks if "predictions.json" in b or "REPRO_RESULT" in b
    ] or blocks
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
    generated.update(p.name for p in workdir.glob("*_probe_trace.md"))
    generated.update(p.name for p in workdir.glob("*_transcript.jsonl"))
    generated.update({"runtime_probes.json", "runtime_probes.sh"})
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
    allow_runtime_probe: bool = False,
    max_runtime_probes_per_role: int = MAX_RUNTIME_PROBES_PER_ROLE,
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
    probe_trace_path = workdir / f"{name}_probe_trace.md"
    probes: list[dict[str, Any]] = []
    submission_trace: str | None = None
    runtime_probe_required = (
        allow_runtime_probe
        and name.startswith("repair_")
        and trigger == "execution_error_and_reviewer_finding"
        and bool(
            re.search(
                r"ModuleNotFoundError|ImportError|TypeError|FileNotFoundError|"
                r"No such file or directory|Dataset not found or corrupted",
                context,
            )
        )
    )

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

    def save_probe_trace() -> None:
        sections = [
            f"# {name} restricted runtime probe trace",
            "",
            "These diagnostics are separate from verifier-visible evaluation commands.",
        ]
        for index, probe in enumerate(probes, 1):
            sections.extend(
                [
                    f"\n## Probe {index}: {probe['kind']} `{probe['target']}`",
                    f"\n```bash\n{probe['command']}\n```",
                    f"\n```text\n{probe['observation']}\n```",
                ]
            )
        text = "\n".join(sections) + "\n"
        for d in (workdir, artifact_dir):
            d.mkdir(parents=True, exist_ok=True)
            (d / probe_trace_path.name).write_text(text)

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

    def runtime_probe(arguments: dict) -> str:
        if not allow_runtime_probe:
            raise ValueError("runtime probes are disabled for this condition")
        role_probe_limit = 1 if runtime_probe_required else max_runtime_probes_per_role
        if len(probes) >= role_probe_limit:
            raise ValueError("runtime probe budget exhausted for this role")
        probe_transcript = getattr(session, "probe_transcript", None)
        probe_fn = getattr(session, "probe", None)
        if probe_transcript is None or probe_fn is None:
            raise ValueError("session does not support separated runtime probes")
        if len(probe_transcript) >= MAX_RUNTIME_PROBES:
            raise ValueError("runtime probe budget exhausted for this run")
        if (
            not runtime_probe_required
            and len(probe_transcript) >= MAX_RUNTIME_PROBES - MAX_REPAIR_ROUNDS
        ):
            raise ValueError(
                "optional runtime probe budget exhausted; remaining probes are "
                "reserved for execution-driven repairs"
            )
        kind = str(arguments.get("kind", "")).strip()
        target = str(arguments.get("target", "")).strip()
        command = _runtime_probe_command(kind, target)
        run = probe_fn(command, timeout=30)
        observation = _runtime_probe_observation(run)
        probes.append(
            {
                "kind": kind,
                "target": target,
                "command": command,
                "observation": observation,
            }
        )
        save_probe_trace()
        return observation

    def submit(arguments: dict) -> str:
        nonlocal submitted
        if not queries:
            raise ValueError("call search_repo with your own query before submitting")
        if runtime_probe_required and not probes:
            raise ValueError(
                "call runtime_probe on the concrete runtime uncertainty before "
                "submitting this repair"
            )
        raw = (
            submission_adapter(arguments)
            if submission_adapter is not None
            else str(arguments.get("content", ""))
        )
        content = validator(raw)
        _atomic_write_text(output_path, content)
        save_submission(raw)
        submitted = True
        return f"accepted and wrote {output_path.name}"

    execution_feedback_trigger = trigger in {
        "execution_result",
        "repair_execution_result",
        "execution_error_and_reviewer_finding",
    }
    action_nudge = (
        (
            "Search the exact exception symbol or failing source path, then call "
            "runtime_probe on the concrete runtime uncertainty before submitting "
            "the repair."
            if runtime_probe_required
            else "Search the exact exception symbol, failing source path, or disputed "
            "API from the latest execution evidence before submitting the artifact."
        )
        if execution_feedback_trigger
        else f"Call search_repo with a query derived from the current context, "
        f"or call {submit_name} when the artifact is grounded and complete."
    )

    tool_schemas = [SEARCH_REPO_TOOL]
    tool_handlers = {"search_repo": dynamic_search, submit_name: submit}
    if allow_runtime_probe:
        tool_schemas.append(RUNTIME_PROBE_TOOL)
        tool_handlers["runtime_probe"] = runtime_probe
    tool_schemas.append(submit_schema or _submit_tool(submit_name, submit_description))

    result = run_agent(
        task,
        session,
        role_llm,
        max_steps=max_steps,
        compress=False,
        use_tools=True,
        system_prompt=instruction,
        initial_user_message=context,
        action_nudge=action_nudge,
        tool_schemas=tool_schemas,
        tool_handlers=tool_handlers,
        stop_when=lambda: submitted or (
            len(queries) >= max_queries
            and (not runtime_probe_required or bool(probes))
        ),
        stop_summary=f"{name} search phase complete",
    )

    synthesis_steps = synthesis_peak = 0
    if queries and not submitted and (not runtime_probe_required or probes):
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
            _atomic_write_text(output_path, validated)
            save_submission(candidate)
            submitted = True
            break
        _save_messages(f"{name}_synthesis", synthesis_messages, workdir, artifact_dir)

    _save_role_transcript(name, result, workdir, artifact_dir)
    if not queries:
        raise RuntimeError(f"{name} submitted no runtime-generated RAG query")
    if runtime_probe_required and not probes:
        raise RuntimeError(f"{name} submitted no required restricted runtime probe")
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
        "runtime_probes": len(probes),
        "runtime_probe_required": runtime_probe_required,
        "probe_trace": probe_trace_path.name if probes else None,
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
    # Task-agnostic sanity diagnostics that are SAFE under generic prompt mode:
    # each is derivable from the agent's own output + the repo's own files, never
    # the hidden target (e.g. "your metric is below random chance — direction is
    # inverted", "your hardcoded normalization disagrees with the repo source").
    # generic mode appends these to the shape-only diagnostics; specialized mode
    # already folds them into public_contract_diagnostics.
    generic_safe_diagnostics: Callable[[Any], list[str]] = field(default=lambda s: [])
    # Random-chance floor for a higher-is-better metric (e.g. 50.0 for binary
    # AUROC, 100/num_classes for balanced top-1 accuracy). When set, the generic
    # path emits a framework-level "below chance ⇒ inverted direction" diagnostic
    # from the verifier-recomputed value — task-agnostic, never the hidden target.
    # None disables the check (e.g. metrics with no meaningful chance baseline).
    chance_level: float | None = None

    # Verify kwargs (forwarded to verify_run)
    verify_kwargs: dict = field(default_factory=dict)

    # Public machine-readable artifact contract. Generic roles stay shared, while
    # the benchmark still tells them what output the external verifier accepts.
    public_result_protocol: str = ""
    # Exact public command used by the orchestrator to invoke the generated
    # program. This is interface information, not an oracle solution hint.
    public_execution_command: str = ""

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

class _PipelineDone(Exception):
    """Clean early-stop for the solo/team ablation conditions (not an error)."""


_PIPELINES = ("solo", "team", "solo-retry", "solo-repair", "full")
_PROMPT_MODES = ("specialized", "generic")
# Every post-execution loop shares ONE execution budget so the conditions are
# comparable: 1 initial run + up to MAX_REPAIR_ROUNDS follow-ups = 5 executions.
MAX_REPAIR_ROUNDS = 4


def _role_prompts(config: OracleConfig, prompt_mode: str) -> RolePrompts:
    if prompt_mode == "generic":
        return GENERIC_PROMPTS
    return RolePrompts(
        navigator=config.navigator_instruction,
        reproducer=config.reproducer_instruction,
        critic=config.critic_instruction,
        reviewer=config.reviewer_instruction,
        repair=config.repair_instruction,
    )


def _generic_task_context(config: OracleConfig) -> str:
    """Expose the public task and verifier interface without oracle solution hints."""
    lines = [
        config.task.strip(),
        "",
        "# Public execution interface",
        (
            f"The orchestrator will invoke the generated program as:\n"
            f"`{config.public_execution_command.strip()}`\n"
            "The program must accept and honor this command's arguments and "
            "provisioned paths."
            if config.public_execution_command.strip()
            else (
                f"The orchestrator will invoke `{config.eval_script}` directly. "
                "Do not require undocumented arguments."
            )
        ),
        "",
        "# Public result protocol",
    ]
    if config.public_result_protocol.strip():
        lines.append(config.public_result_protocol.strip())
        lines.extend(
            [
                "",
                "The verifier accepts only this artifact contract. Generate it from",
                "the real evaluation; printed aggregate metrics are not evidence.",
            ]
        )
        return "\n".join(lines)

    # Backward-compatible contract for any legacy aggregate-result oracle.
    lines.extend(
        [
            "A result counts only when a successful real evaluation command prints one",
            "strict-JSON line beginning with `REPRO_RESULT `.",
            f'The JSON metric id must be "{config.metric}".',
            "The JSON `actual` value must use the units requested by the public task.",
        ]
    )
    expected_n = config.verify_kwargs.get("expected_num_examples")
    if expected_n is not None:
        lines.append(f"The JSON `num_examples` value must be {expected_n}.")
    expected_datasets = config.verify_kwargs.get("expected_datasets")
    if expected_datasets is not None:
        lines.append(
            "Include evaluated dataset counts in `datasets` for: "
            + ", ".join(str(name) for name in expected_datasets)
            + "."
        )
    expected_runs = config.verify_kwargs.get("expected_runs")
    if expected_runs is not None:
        lines.append(
            "Include per-run, per-dataset measured values in `run_metrics` for: "
            + ", ".join(str(name) for name in expected_runs)
            + "."
        )
    expected_aggregation = config.verify_kwargs.get("expected_aggregation")
    if expected_aggregation is not None:
        lines.append(f'Use aggregation identifier "{expected_aggregation}".')
    lines.append(
        "The evaluation program must print this line from its measured output; "
        "do not echo, relay, or hardcode a result."
    )
    return "\n".join(lines)


def _make_generic_code_validator(config: OracleConfig) -> Callable[[str], str]:
    """Validate only syntax and the public artifact interface, not task solutions."""
    artifact_markers = sorted(
        set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol))
    )
    if not config.public_result_protocol.strip():
        artifact_markers = ["REPRO_RESULT"]

    def validate(content: str) -> str:
        code = _extract_python(content)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"code is not syntactically valid: {exc}") from exc
        missing = [marker for marker in artifact_markers if marker not in code]
        if missing:
            raise ValueError(
                "code does not produce the public result artifact described by the "
                f"runtime contract (missing: {missing})"
            )
        return code

    return validate


def _failed_import_packages(session: Any, workdir: Path, start: int = 0) -> set[str]:
    """Extract workspace package initializers implicated by public import failures."""
    failed: set[str] = set()
    workspace = workdir.resolve()
    for run in session.transcript[start:]:
        log = f"{run.stdout}\n{run.stderr}"
        if "ModuleNotFoundError" not in log and "ImportError" not in log:
            continue
        for raw_path in re.findall(r'File "([^"]+/__init__\.py)"', log):
            path = Path(raw_path)
            if raw_path.startswith("/workspace/"):
                relative = Path(raw_path.removeprefix("/workspace/"))
            elif path.is_absolute():
                try:
                    relative = path.resolve().relative_to(workspace)
                except ValueError:
                    continue
            else:
                relative = path
            package_parts = relative.parent.parts
            if package_parts and all(part.isidentifier() for part in package_parts):
                failed.add(".".join(package_parts))
    return failed


def _make_generic_repair_validator(
    base_validator: Callable[[str], str],
    session: Any,
    workdir: Path,
    execution_start: int,
    current_code: str | None = None,
) -> Callable[[str], str]:
    """Prevent a repair from re-entering package initializers proven to fail."""
    failed_packages = _failed_import_packages(session, workdir, execution_start)

    def validate(content: str) -> str:
        code = base_validator(content)
        if current_code is not None and code.strip() == current_code.strip():
            raise ValueError(
                "repair made no code change after a failed execution; address the "
                "current blocker before resubmitting"
            )
        if not failed_packages:
            return code
        tree = ast.parse(code)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        reentered = sorted(
            package
            for package in failed_packages
            if any(name == package or name.startswith(package + ".") for name in imports)
        )
        if reentered:
            raise ValueError(
                "repair re-enters package initializers already proven to fail by "
                f"the public execution traceback: {reentered}. Use a different "
                "repository-grounded path instead of a sibling submodule."
            )
        return code

    return validate


def _below_chance_diagnostic(
    actual: float, chance_level: float, metric: str = "metric"
) -> str | None:
    """Framework-level sanity check, safe under generic mode: a higher-is-better
    metric recomputed below its random-chance floor implies an inverted score or
    decision direction. References only the agent's own recomputed value and the
    config-declared floor — never the hidden target."""
    if actual >= chance_level:
        return None
    return (
        f"The recomputed {metric} ({actual}) is below the {chance_level} "
        f"random-chance baseline for this higher-is-better metric. A real method "
        f"scoring below chance indicates an inverted score or label/decision "
        f"direction — correct the scoring/decision polarity so the metric exceeds "
        f"chance; do not simply negate the reported number."
    )


def _make_generic_contract_diagnostics(
    config: OracleConfig, pass_gate: Callable[[Any], bool] | None = None
) -> Callable[[Any], list[str]]:
    """Expose pass/fail contract feedback without oracle-specific repair hints.

    ``pass_gate`` decides the early "已通过, 无反馈" return. It defaults to
    ``config.public_contract_passes`` (preserving direct-call behaviour), but
    ``run_oracle`` injects a *framework-level* gate in generic mode so that no
    oracle-specific check (e.g. a normalization comparison) silently referees the
    pass decision — such checks may still appear as feedback, but never as the
    gate.
    """
    gate = pass_gate if pass_gate is not None else config.public_contract_passes
    artifact_markers = sorted(
        set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol))
    )

    def json_shape(value: Any, depth: int = 0) -> str:
        if isinstance(value, list):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            if depth >= 2:
                return f"dict[{len(value)} keys]"
            items = list(value.items())[:12]
            body = ", ".join(
                f"{key}: {json_shape(child, depth + 1)}" for key, child in items
            )
            suffix = ", ..." if len(value) > len(items) else ""
            return "{" + body + suffix + "}"
        return type(value).__name__

    def diagnostics(session: Any) -> list[str]:
        if gate(session):
            return []
        missing = [
            marker for marker in artifact_markers if not (config.workdir / marker).is_file()
        ]
        if missing:
            return [
                "The required public result artifact is missing after execution "
                f"(missing: {missing}). Inspect the public task, result protocol, "
                "and execution log."
            ]
        observations: list[str] = []
        for marker in artifact_markers:
            path = config.workdir / marker
            if path.suffix == ".json":
                try:
                    observations.append(f"{marker}: {json_shape(json.loads(path.read_text()))}")
                except (OSError, ValueError):
                    observations.append(f"{marker}: invalid JSON")
        recompute = config.verify_kwargs.get("recompute_fn")
        measured = None
        if callable(recompute):
            try:
                measured = recompute(config.workdir)
            except Exception:
                measured = None
            if (
                isinstance(measured, tuple)
                and len(measured) >= 2
                and isinstance(measured[0], (int, float))
            ):
                observations.append(
                    f"public verifier recomputed {config.metric}={measured[0]} "
                    f"over n={measured[1]} from this artifact"
                )
        observed = (
            " Observed public artifact evidence: " + "; ".join(observations) + "."
            if observations
            else ""
        )
        base = [
            "The public result artifact exists but the deterministic verifier "
            "rejected it as malformed, incomplete, or semantically invalid. "
            "Inspect the public result protocol, repository source, and execution log."
            + observed
        ]
        # Framework-level below-chance check: a recomputed higher-is-better metric
        # under its random-chance floor implies an inverted score/decision
        # direction. Task-agnostic (floor comes from config.chance_level, value
        # from the verifier's own recomputation) — never the hidden target.
        if (
            config.chance_level is not None
            and isinstance(measured, tuple)
            and len(measured) >= 1
            and isinstance(measured[0], (int, float))
        ):
            below = _below_chance_diagnostic(measured[0], config.chance_level, config.metric)
            if below:
                base.append(below)
        # Append any oracle-specific safe diagnostics (e.g. code-vs-repo
        # normalization). Safe under generic mode: they reference only the agent's
        # own output and the repo's own files, never the hidden target.
        try:
            base.extend(config.generic_safe_diagnostics(session) or [])
        except Exception:
            pass
        return base

    return diagnostics


def run_oracle(
    config: OracleConfig,
    pipeline: str = "full",
    prompt_mode: str = "specialized",
) -> None:
    """Run one blind reproduction under an ablation condition.

    The five conditions share an identical execution budget (≤5 evals); they
    differ only in *what* drives the follow-up attempts — which is what isolates
    "another attempt" from "repair driven by the real error":

      * ``"solo"``        — Reproducer only → 1 execution. Single-agent baseline.
      * ``"team"``        — Navigator + Reproducer + Critic → 1 execution.
                            Pre-execution collaboration, no follow-ups.
      * ``"solo-retry"``  — Reproducer; on failure RE-GENERATE from the original
                            context with **no execution feedback**, ≤5 executions.
                            The budget-matched control for "more tries".
      * ``"solo-repair"`` — Reproducer; on failure a Repair role fixes it **with
                            the real execution error**, ≤5 executions. Single
                            agent + feedback repair.
      * ``"full"``        — Navigator + Reproducer + Critic + Reviewer + feedback
                            Repair loop, ≤5 executions (default).

    Non-``full`` runs write artifacts to a ``__<pipeline>`` dir so conditions
    never collide.
    """
    if pipeline not in _PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; valid: {_PIPELINES}")
    if prompt_mode not in _PROMPT_MODES:
        raise ValueError(f"unknown prompt mode {prompt_mode!r}; valid: {_PROMPT_MODES}")
    prompts = _role_prompts(config, prompt_mode)
    generic_mode = prompt_mode == "generic"
    task_context = (
        _generic_task_context(config) if generic_mode else config.task
    )
    code_validator = (
        _make_generic_code_validator(config) if generic_mode else config.validate_code
    )
    # Generic pass gate — framework-level, with NO oracle-specific check (so a
    # normalization comparison etc. can be feedback but never the referee). When
    # the task is on V2 recomputation, "pass" = artifact present + recomputable +
    # (if a chance floor is declared) at or above it. Tasks without a recompute_fn
    # fall back to the oracle gate (preserves non-V2 / mock behaviour).
    def _generic_pass_gate(session: Any) -> bool:
        recompute_fn = config.verify_kwargs.get("recompute_fn")
        if not callable(recompute_fn):
            return config.public_contract_passes(session)
        markers = sorted(
            set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol))
        )
        if markers and not all((config.workdir / m).is_file() for m in markers):
            return False
        try:
            probe = recompute_fn(config.workdir)
        except Exception:
            probe = None
        if not (isinstance(probe, tuple) and probe and isinstance(probe[0], (int, float))):
            return False
        if config.chance_level is not None and probe[0] < config.chance_level:
            return False
        return True

    generic_pass_gate = _generic_pass_gate if generic_mode else config.public_contract_passes
    contract_diagnostics = (
        _make_generic_contract_diagnostics(config, pass_gate=generic_pass_gate)
        if generic_mode
        else config.public_contract_diagnostics
    )
    repair_submit_name = "submit_code" if generic_mode else config.repair_submit_name
    repair_submit_description = (
        f"Submit the complete repaired {config.eval_script}."
        if generic_mode
        else config.repair_submit_description
    )
    repair_submit_schema = None if generic_mode else config.repair_submit_schema
    repair_submission_adapter = (
        None if generic_mode else config.repair_submission_adapter
    )
    repair_synthesis_instruction = (
        None if generic_mode else config.repair_synthesis_instruction
    )
    generic_code_synthesis_instruction = (
        f"Return only the complete executable source code for {config.eval_script}. "
        "The program must produce the public result artifact when executed. "
        "Do not return the contents of predictions or result files."
        if generic_mode
        else None
    )
    run_critic = pipeline in ("team", "full")
    post_mode = {
        "solo": "none", "team": "none",
        "solo-retry": "retry", "solo-repair": "repair", "full": "repair",
    }[pipeline]
    use_reviewer = pipeline == "full"

    artifact_suffixes = []
    if prompt_mode != "specialized":
        artifact_suffixes.append(prompt_mode)
    if pipeline != "full":
        artifact_suffixes.append(pipeline)
    artifact_dir = config.artifact_dir
    if artifact_suffixes:
        artifact_dir = config.artifact_dir.parent / (
            f"{config.artifact_dir.name}__{'__'.join(artifact_suffixes)}"
        )

    config.copy_clean_source()
    if config.assert_blind_workspace is not None:
        config.assert_blind_workspace()
    for pattern in ("*_probe_trace.md", "runtime_probes.json", "runtime_probes.sh"):
        for generated_path in config.workdir.glob(pattern):
            generated_path.unlink(missing_ok=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    session = config.make_session()
    if config.session_go_offline:
        session.go_offline()

    def rag_role(**kwargs: Any) -> tuple[dict, dict]:
        return _dynamic_rag_role(
            task=config.task,
            workdir=config.workdir,
            artifact_dir=artifact_dir,
            session=session,
            search_extra_exclude=config.search_extra_exclude,
            allow_runtime_probe=generic_mode,
            **kwargs,
        )

    roles: dict[str, dict] = {}
    rag: dict[str, dict] = {}
    protected_code_blocks: set[str] = set()
    workflow_error: str | None = None
    execution_start = 0
    n_exec = 0  # actual eval executions consumed (budget reporting)

    def sync_eval_file() -> None:
        sync_file = getattr(session, "sync_file", None)
        if sync_file is not None and not sync_file(config.eval_script):
            raise RuntimeError(
                f"generated evaluation file is not visible to the execution session: "
                f"{config.eval_script}"
            )

    try:
        if run_critic:  # Navigator runs whenever the pre-execution team runs.
            roles["navigator"], rag["navigator"] = rag_role(
                name="navigator",
                instruction=prompts.navigator,
                context=task_context,
                output_path=config.workdir / "navigator_report.md",
                submit_name="submit_handoff",
                submit_description="Submit the source-grounded Navigator handoff.",
                validator=config.validate_report,
                trigger="initial_task",
                max_steps=7,
            )
            builder_context = (
                (
                    "# Public task and result protocol\n\n"
                    + task_context
                    + "\n\n"
                    if generic_mode
                    else ""
                )
                + "# Navigator handoff\n\n"
                + (config.workdir / "navigator_report.md").read_text(errors="replace")
            )
        else:  # solo: the Reproducer works straight from the task.
            builder_context = task_context

        roles["reproducer"], rag["reproducer"] = rag_role(
            name="reproducer",
            instruction=prompts.reproducer,
            context=builder_context,
            output_path=config.workdir / config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete generated {config.eval_script}.",
            validator=code_validator,
            trigger="navigator_handoff" if run_critic else "initial_task",
            max_steps=7,
            synthesis_instruction=generic_code_synthesis_instruction,
            synthesis_attempts=5,
        )

        if run_critic:
            critic_context = (
                (
                    "# Public task and result protocol\n\n"
                    + task_context
                    + "\n\n"
                    if prompt_mode == "generic"
                    else ""
                )
                + "# Generated evaluation script\n\n"
                + (config.workdir / config.eval_script).read_text(errors="replace")
                + "\n\n# Navigator handoff\n\n"
                + (config.workdir / "navigator_report.md").read_text(errors="replace")
            )
            roles["critic"], rag["critic"] = rag_role(
                name="critic",
                instruction=prompts.critic,
                context=critic_context,
                output_path=config.workdir / config.eval_script,
                submit_name="submit_code",
                submit_description=f"Submit the complete audited {config.eval_script}.",
                validator=code_validator,
                trigger="generated_code_audit",
                max_steps=7,
                synthesis_instruction=generic_code_synthesis_instruction,
                synthesis_attempts=5,
            )

        sync_eval_file()
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
        latest_execution_start = execution_start

        def review_current(round_index: int) -> None:
            diagnostics = contract_diagnostics(session)
            review_log_start = latest_execution_start if generic_mode else execution_start
            review_context = (
                (
                    "# Public task and result protocol\n\n"
                    + task_context
                    + "\n\n"
                    if prompt_mode == "generic"
                    else ""
                )
                + "# Navigator handoff\n\n"
                + (config.workdir / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Evaluation implementation\n\n"
                + _clip(
                    (config.workdir / config.eval_script).read_text(errors="replace"),
                    12000,
                )
                + (
                    "\n\n# Latest public execution log\n\n"
                    if generic_mode
                    else "\n\n# Public execution logs\n\n"
                )
                + _clip(
                    _public_log(session, review_log_start),
                    12000,
                )
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"reviewer_{round_index}"
            roles[key], rag[key] = rag_role(
                name=key,
                instruction=prompts.reviewer,
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

        n_exec = 1  # the initial execution above
        if post_mode == "none":
            raise _PipelineDone()  # solo/team stop after the first execution

        if use_reviewer:
            review_current(0)

        for round_index in range(1, MAX_REPAIR_ROUNDS + 1):
            if generic_pass_gate(session):
                break
            diagnostics = contract_diagnostics(session)
            accepted_new_blocks: list[str] = []

            if post_mode == "retry":
                # Budget-matched CONTROL: regenerate from the ORIGINAL context with
                # NO execution feedback — isolates "another attempt" from "repair
                # driven by the real error". Same execution budget as repair/full.
                key = f"retry_{round_index}"
                roles[key], rag[key] = rag_role(
                    name=key,
                    instruction=prompts.reproducer,
                    context=builder_context,
                    output_path=config.workdir / config.eval_script,
                    submit_name="submit_code",
                    submit_description=f"Submit a fresh {config.eval_script}.",
                    validator=code_validator,
                    trigger="blind_retry",
                    max_steps=7,
                    synthesis_instruction=generic_code_synthesis_instruction,
                    synthesis_attempts=5,
                )
            else:  # "repair": fix WITH the real execution error (solo-repair, full)
                parts = []
                if prompt_mode == "generic":
                    parts.append("# Public task and result protocol\n\n" + task_context)
                parts.extend(
                    [
                        "# Current evaluation script\n\n"
                        + (config.workdir / config.eval_script).read_text(errors="replace"),
                        (
                            "# Latest public execution log\n\n"
                            + _public_log(session, latest_execution_start)
                            if generic_mode
                            else "# Public execution log\n\n"
                            + _public_log(session, execution_start)
                        ),
                    ]
                )
                if generic_mode and latest_execution_start != execution_start:
                    parts.append(
                        "# Prior execution history (clipped)\n\n"
                        + _clip(_public_log(session, execution_start), 6000)
                    )
                if use_reviewer:
                    parts.append(
                        "# Independent reviewer audit\n\n"
                        + (config.workdir / "review_report.md").read_text(errors="replace")
                    )
                if run_critic:
                    parts.append(
                        "# Navigator handoff\n\n"
                        + (config.workdir / "navigator_report.md").read_text(errors="replace")
                    )
                parts.append(
                    "# Deterministic public-contract audit\n\n"
                    + "\n".join(f"- {issue}" for issue in diagnostics)
                )
                repair_context = "\n\n".join(parts)
                if generic_mode:
                    repair_validator = _make_generic_repair_validator(
                        code_validator,
                        session,
                        config.workdir,
                        execution_start,
                        current_code=(
                            config.workdir / config.eval_script
                        ).read_text(errors="replace"),
                    )
                elif config.repair_make_validator is not None:
                    repair_validator = config.repair_make_validator(
                        diagnostics, protected_code_blocks, accepted_new_blocks
                    )
                else:
                    repair_validator = code_validator
                key = f"repair_{round_index}"
                roles[key], rag[key] = rag_role(
                    name=key,
                    # .replace (not .format): repair instructions embed literal JSON
                    # braces from EVIDENCE, which str.format would mis-parse as fields.
                    instruction=prompts.repair.replace(
                        "{round_index}", str(round_index)
                    ),
                    context=repair_context,
                    output_path=config.workdir / config.eval_script,
                    submit_name=repair_submit_name,
                    submit_description=repair_submit_description,
                    validator=repair_validator,
                    trigger="execution_error_and_reviewer_finding",
                    max_steps=7,
                    max_queries=3 if generic_mode else 2,
                    submit_schema=repair_submit_schema,
                    submission_adapter=repair_submission_adapter,
                    synthesis_instruction=(
                        generic_code_synthesis_instruction
                        if generic_mode
                        else repair_synthesis_instruction
                    ),
                    synthesis_attempts=4,
                )

            sync_eval_file()
            start = len(session.transcript)
            stepped_run = config.execute_eval(session)
            n_exec += 1
            latest_execution_start = start
            roles[key]["errors"] = 0 if stepped_run.ok else 1
            roles[key]["command_indexes"] = [start + 1, len(session.transcript)]
            session.write_file(
                "reproducer_public_log.txt",
                _public_log(session, execution_start),
            )

            if use_reviewer:
                review_current(round_index)
                if config.make_endorsed is not None and config.make_endorsed(
                    stepped_run.ok,
                    generic_pass_gate(session),
                    config.workdir / "review_report.md",
                ):
                    protected_code_blocks.update(accepted_new_blocks)

    except _PipelineDone:
        pass  # solo/team intentionally stop after the first execution
    except Exception as exc:
        workflow_error = f"{type(exc).__name__}: {exc}"
    finally:
        # Docker-backed sessions need teardown; the subprocess Session has no
        # close() (its state is just the workdir), so call it only if present.
        close = getattr(session, "close", None)
        if close is not None:
            close()

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
    # Handoff artifacts required scale with the pipeline: full needs the reviewer
    # audit, team needs the navigator handoff, solo has neither.
    handoff_requirement = True
    if run_critic:
        handoff_requirement = (config.workdir / "navigator_report.md").exists()
    if use_reviewer:
        handoff_requirement = handoff_requirement and (
            config.workdir / "review_report.md"
        ).exists()
    collaboration_pass = verdict.match and rag_requirement and handoff_requirement
    total_cost = round(
        sum(r["usage"].get("cost_yuan", 0.0) for r in roles.values())
        + sum(s["usage"].get("cost_yuan", 0.0) for s in rag.values()),
        4,
    )
    probe_transcript = list(getattr(session, "probe_transcript", []))
    output = {
        "task": config.task,
        "pipeline": pipeline,
        "prompt_mode": prompt_mode,
        "oracle_contract_mode": (
            "public_artifact_only" if generic_mode else "task_specific"
        ),
        "max_executions": MAX_REPAIR_ROUNDS + 1,  # shared budget across conditions
        "eval_executions": n_exec,                # actually consumed
        "blind_workspace_checked": config.assert_blind_workspace is not None,
        "agents": len(roles),
        "attempt": config.attempt,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "retrieval_ranker": config.retrieval_ranker,
        "repair_mode": (
            "full_file_replacement" if generic_mode else config.repair_mode_label
        ),
        "workflow_error": workflow_error,
        "total_rag_calls": sum(stage["calls"] for stage in rag.values()),
        "rag_requirement_met": rag_requirement,
        "handoff_requirement_met": handoff_requirement,
        "public_evidence_found": generic_pass_gate(session),
        "public_contract_diagnostics": contract_diagnostics(session),
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": len(session.transcript),
        "runtime_probe_enabled": generic_mode,
        "runtime_probe_budget": MAX_RUNTIME_PROBES if generic_mode else 0,
        "total_runtime_probes": len(probe_transcript),
    }
    result_json = json.dumps(output, indent=2) + "\n"

    replay_fn = getattr(session, "replay_script", None)
    replay_script = (replay_fn() + "\n") if replay_fn is not None else None
    probe_replay_fn = getattr(session, "probe_replay_script", None)
    probe_replay_script = (
        (probe_replay_fn() + "\n") if probe_replay_fn is not None and probe_transcript else None
    )
    probe_json = json.dumps(
        [
            {
                "command": run.command,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "duration_s": run.duration_s,
            }
            for run in probe_transcript
        ],
        indent=2,
    ) + "\n"

    for output_dir in (config.workdir, artifact_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(result_json)
        if replay_script is not None:
            (output_dir / "commands.sh").write_text(replay_script)
        if probe_replay_script is not None:
            (output_dir / "runtime_probes.sh").write_text(probe_replay_script)
            (output_dir / "runtime_probes.json").write_text(probe_json)
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
            for trace in config.workdir.glob("*_probe_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)

    print(result_json)
