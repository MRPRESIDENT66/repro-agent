"""Dynamic role loop helpers for repository navigation and synthesis."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from agent.llm import ChatLLM
from agent.loop import AgentResult, TOOLS, run_agent
from agent.runtime_probe import (
    MAX_RUNTIME_PROBES,
    MAX_RUNTIME_PROBES_PER_ROLE,
    RUNTIME_PROBE_TOOL,
    runtime_probe_command as _runtime_probe_command,
    runtime_probe_observation as _runtime_probe_observation,
)
from retrieval.search import relevant_snippet, search_repo

SEARCH_REPO_TOOL = next(t for t in TOOLS if t["function"]["name"] == "search_repo")
MAX_REPAIR_ROUNDS = 4


@dataclass(frozen=True)
class RoleDeps:
    llm_factory: Callable[[], ChatLLM] = ChatLLM
    search_fn: Callable[..., str] = search_repo
    snippet_fn: Callable[..., str] = relevant_snippet


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


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish generated code atomically for Docker bind-mount readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _require_handoff(path: Path, name: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"{name} handoff missing: {path.name}")
    return path.read_text(errors="replace")


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
    deps: RoleDeps | None = None,
) -> str:
    deps = deps or RoleDeps()
    generated = set(extra_exclude or ())
    generated.update(p.name for p in workdir.glob("*_rag_trace.md"))
    generated.update(p.name for p in workdir.glob("*_probe_trace.md"))
    generated.update(p.name for p in workdir.glob("*_transcript.jsonl"))
    generated.update({"runtime_probes.json", "runtime_probes.sh"})
    ranking_evidence = _search_evidence(context or "")
    path_hints = _missing_path_hints(context or "", workdir)
    if path_hints:
        ranking_evidence += "\nExisting files beside the missing path:\n" + "\n".join(path_hints)
    result = deps.search_fn(
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
                f"{deps.snippet_fn(p, f'{query}\n{ranking_evidence}', 3200)}"
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
    synthesis_validator: Callable[[str], str] | None = None,
    synthesis_attempts: int = 3,
    allow_runtime_probe: bool = False,
    max_runtime_probes_per_role: int = MAX_RUNTIME_PROBES_PER_ROLE,
    deps: RoleDeps | None = None,
) -> tuple[dict, dict]:
    deps = deps or RoleDeps()
    role_llm = deps.llm_factory()
    rag_llm = deps.llm_factory()
    synthesis_llm = deps.llm_factory()
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
    suggested_probe_match = re.search(
        r"(?m)^-\s*suggested_probe:\s*(\S+)\s*$",
        context,
    )
    runtime_probe_recommended = (
        allow_runtime_probe
        and name.startswith("repair_")
        and trigger == "execution_error_and_reviewer_finding"
        and suggested_probe_match is not None
    )
    suggested_probe = suggested_probe_match.group(1) if suggested_probe_match else None

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
            deps=deps,
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
        role_probe_limit = max_runtime_probes_per_role
        if len(probes) >= role_probe_limit:
            raise ValueError("runtime probe budget exhausted for this role")
        probe_transcript = getattr(session, "probe_transcript", None)
        probe_fn = getattr(session, "probe", None)
        if probe_transcript is None or probe_fn is None:
            raise ValueError("session does not support separated runtime probes")
        if len(probe_transcript) >= MAX_RUNTIME_PROBES:
            raise ValueError("runtime probe budget exhausted for this run")
        if (
            not runtime_probe_recommended
            and len(probe_transcript) >= MAX_RUNTIME_PROBES - MAX_REPAIR_ROUNDS
        ):
            raise ValueError(
                "optional runtime probe budget exhausted; remaining probes are "
                "reserved for failure-classifier suggested probes"
            )
        kind = str(arguments.get("kind", "")).strip()
        target = str(arguments.get("target", "")).strip()
        command = _runtime_probe_command(kind, target)
        run = probe_fn(command, timeout=30)
        observation = _runtime_probe_observation(run, _clip)
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
            "The failure classifier suggests runtime_probe "
            f"`{suggested_probe}`. Use it if repository evidence is insufficient; "
            "you may submit without probing when the repair is already grounded."
            if runtime_probe_recommended
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
        ),
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
                validated = (synthesis_validator or validator)(candidate)
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
    if not submitted:
        raise RuntimeError(f"{name} failed to synthesize a valid artifact")

    role = {
        "steps": result.steps + synthesis_steps,
        "errors": result.errors,
        "format_errors": result.format_errors,
        "gave_final": submitted,
        "usage": {
            "llm_calls": role_llm.usage.as_dict()["llm_calls"] + synthesis_llm.usage.as_dict()["llm_calls"],
            "prompt_tokens": role_llm.usage.as_dict()["prompt_tokens"] + synthesis_llm.usage.as_dict()["prompt_tokens"],
            "cache_hit_tokens": role_llm.usage.as_dict()["cache_hit_tokens"] + synthesis_llm.usage.as_dict()["cache_hit_tokens"],
            "completion_tokens": role_llm.usage.as_dict()["completion_tokens"] + synthesis_llm.usage.as_dict()["completion_tokens"],
            "cost_yuan": round(role_llm.usage.as_dict()["cost_yuan"] + synthesis_llm.usage.as_dict()["cost_yuan"], 4),
        },
        "peak_ctx_tokens": max(result.peak_ctx_tokens, synthesis_peak),
        "tool_counts": result.tool_counts,
        "command_indexes": [],
        "submission_trace": submission_trace,
        "runtime_probes": len(probes),
        "runtime_probe_required": False,
        "runtime_probe_recommended": runtime_probe_recommended,
        "runtime_probe_hint": suggested_probe,
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
