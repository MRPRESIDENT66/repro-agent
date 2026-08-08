"""Dynamic role loop helpers for repository navigation and synthesis."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from agent.runtime.llm import ChatLLM
from agent.runtime.loop import AgentResult, run_agent
from agent.runtime.runtime_probe import (
    MAX_RUNTIME_PROBES,
    MAX_RUNTIME_PROBES_PER_ROLE,
    RUNTIME_PROBE_TOOL,
    runtime_probe_command,
    runtime_probe_observation,
)
from retrieval.search import relevant_snippet, search_repo

SEARCH_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "search_repo",
        "description": "Search the repository for files relevant to the current uncertainty.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
MAX_REPAIR_ROUNDS = 4


@dataclass(frozen=True)
class RoleDeps:
    llm_factory: Callable[[], ChatLLM] = ChatLLM
    search_fn: Callable[..., str] = search_repo
    snippet_fn: Callable[..., str] = relevant_snippet


def clip_text(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


def public_log(session: Any, start: int) -> str:
    parts = []
    for index, run in enumerate(session.transcript[start:], start + 1):
        parts.append(
            f"## Command {index}\n\n```bash\n{run.command}\n```\n\n"
            f"exit={run.exit_code} timed_out={run.timed_out}\n\n"
            f"```text\n{clip_text(run.stdout)}\n{clip_text(run.stderr)}\n```\n"
        )
    return "\n".join(parts)


def atomic_write_text(path: Path, content: str) -> None:
    """Publish generated code atomically for Docker bind-mount readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def require_handoff(path: Path, name: str) -> str:
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


def missing_path_hints(context: str, workdir: Path) -> list[str]:
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
    path_hints = missing_path_hints(context or "", workdir)
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


@dataclass
class _RoleTools:
    """State and tool handlers for one role invocation.

    Keeping this state on an object avoids nested functions and ``nonlocal``.
    The LLM-facing tools below are ordinary methods: search, probe, and submit.
    """

    name: str
    workdir: Path
    artifact_dir: Path
    session: Any
    context: str
    output_path: Path
    validator: Callable[[str], str]
    submission_adapter: Callable[[dict], str] | None
    search_extra_exclude: set[str] | None
    max_queries: int
    allow_runtime_probe: bool
    max_runtime_probes: int
    deps: RoleDeps
    rag_llm: ChatLLM
    trigger: str
    queries: list[str] = field(default_factory=list, init=False)
    probes: list[dict[str, Any]] = field(default_factory=list, init=False)
    submitted: bool = field(default=False, init=False)
    submission_trace: str | None = field(default=None, init=False)
    suggested_probe: str | None = field(default=None, init=False)
    runtime_probe_recommended: bool = field(default=False, init=False)
    trace_sections: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.trace_sections = [
            f"# {self.name} dynamic RAG trace",
            "",
            f"Trigger: {self.trigger}",
            "",
            "Queries below were generated by the role at runtime.",
        ]
        match = re.search(r"(?m)^-\s*suggested_probe:\s*(\S+)\s*$", self.context)
        self.suggested_probe = match.group(1) if match else None
        self.runtime_probe_recommended = (
            self.allow_runtime_probe
            and self.name.startswith("repair_")
            and self.suggested_probe is not None
        )

    @property
    def trace_path(self) -> Path:
        return self.workdir / f"{self.name}_rag_trace.md"

    @property
    def probe_trace_path(self) -> Path:
        return self.workdir / f"{self.name}_probe_trace.md"

    def _write_to_output_dirs(self, filename: str, content: str) -> None:
        for directory in (self.workdir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / filename).write_text(content)

    def save_trace(self) -> None:
        self._write_to_output_dirs(
            self.trace_path.name, "\n".join(self.trace_sections) + "\n"
        )

    def save_submission(self, raw: str) -> None:
        if self.submission_adapter is None:
            return
        self.submission_trace = f"{self.name}_submission.json"
        self._write_to_output_dirs(self.submission_trace, raw.strip() + "\n")

    def save_probe_trace(self) -> None:
        sections = [
            f"# {self.name} restricted runtime probe trace",
            "",
            "These diagnostics are separate from verifier-visible evaluation commands.",
        ]
        for index, probe in enumerate(self.probes, 1):
            sections.extend(
                [
                    f"\n## Probe {index}: {probe['kind']} `{probe['target']}`",
                    f"\n```bash\n{probe['command']}\n```",
                    f"\n```text\n{probe['observation']}\n```",
                ]
            )
        self._write_to_output_dirs(
            self.probe_trace_path.name, "\n".join(sections) + "\n"
        )

    def search_repository(self, arguments: dict) -> str:
        query = str(arguments.get("query", "")).strip()
        if len(query) < 8:
            raise ValueError("query must describe the current uncertainty")
        if query in self.queries:
            raise ValueError("duplicate query; refine it from the latest evidence")
        if len(self.queries) >= self.max_queries:
            raise ValueError("dynamic RAG query budget exhausted; submit the artifact")

        result = _search_with_snippets(
            query,
            self.rag_llm,
            self.workdir,
            context=self.context,
            extra_exclude=self.search_extra_exclude,
            deps=self.deps,
        )
        self.queries.append(query)
        index = len(self.queries)
        self.trace_sections.extend(
            [f"\n## Query {index}\n\n{query}", f"\n## Result {index}\n\n{result}"]
        )
        self.save_trace()
        return result

    def probe_runtime(self, arguments: dict) -> str:
        if not self.allow_runtime_probe:
            raise ValueError("runtime probes are disabled for this condition")
        if len(self.probes) >= self.max_runtime_probes:
            raise ValueError("runtime probe budget exhausted for this role")

        probe_transcript = self.session.probe_transcript
        if len(probe_transcript) >= MAX_RUNTIME_PROBES:
            raise ValueError("runtime probe budget exhausted for this run")
        reserved = MAX_RUNTIME_PROBES - MAX_REPAIR_ROUNDS
        if not self.runtime_probe_recommended and len(probe_transcript) >= reserved:
            raise ValueError(
                "optional runtime probe budget exhausted; remaining probes are "
                "reserved for failure-classifier suggested probes"
            )

        kind = str(arguments.get("kind", "")).strip()
        target = str(arguments.get("target", "")).strip()
        command = runtime_probe_command(kind, target)
        run = self.session.probe(command, timeout=30)
        observation = runtime_probe_observation(run, clip_text)
        self.probes.append(
            {
                "kind": kind,
                "target": target,
                "command": command,
                "observation": observation,
            }
        )
        self.save_probe_trace()
        return observation

    def submit_artifact(self, arguments: dict) -> str:
        if not self.queries:
            raise ValueError("call search_repo with your own query before submitting")
        raw = (
            self.submission_adapter(arguments)
            if self.submission_adapter is not None
            else str(arguments.get("content", ""))
        )
        content = self.validator(raw)
        atomic_write_text(self.output_path, content)
        self.save_submission(raw)
        self.submitted = True
        return f"accepted and wrote {self.output_path.name}"

    def should_stop(self) -> bool:
        return self.submitted or len(self.queries) >= self.max_queries


class RoleSynthesisError(RuntimeError):
    """A role exhausted artifact synthesis while retaining auditable usage."""

    def __init__(self, name: str, role: dict, rag: dict) -> None:
        super().__init__(f"{name} failed to synthesize a valid artifact")
        self.role = role
        self.rag = rag


def run_rag_role(
    *,
    name: str,
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
    tools = _RoleTools(
        name=name,
        workdir=workdir,
        artifact_dir=artifact_dir,
        session=session,
        context=context,
        output_path=output_path,
        validator=validator,
        submission_adapter=submission_adapter,
        search_extra_exclude=search_extra_exclude,
        max_queries=max_queries,
        allow_runtime_probe=allow_runtime_probe,
        max_runtime_probes=max_runtime_probes_per_role,
        deps=deps,
        rag_llm=rag_llm,
        trigger=trigger,
    )

    has_execution_feedback = trigger in {
        "execution_result",
        "repair_execution_result",
        "execution_error_and_reviewer_finding",
    }
    if not has_execution_feedback:
        action_nudge = (
            "Call search_repo with a query derived from the current context, "
            f"or call {submit_name} when the artifact is grounded and complete."
        )
    elif tools.runtime_probe_recommended:
        action_nudge = (
            "The failure classifier suggests runtime_probe "
            f"`{tools.suggested_probe}`. Use it if repository evidence is "
            "insufficient; you may submit without probing when the repair is "
            "already grounded."
        )
    else:
        action_nudge = (
            "Search the exact exception symbol, failing source path, or disputed "
            "API from the latest execution evidence before submitting the artifact."
        )

    tool_schemas = [SEARCH_REPO_TOOL]
    tool_handlers = {
        "search_repo": tools.search_repository,
        submit_name: tools.submit_artifact,
    }
    if allow_runtime_probe:
        tool_schemas.append(RUNTIME_PROBE_TOOL)
        tool_handlers["runtime_probe"] = tools.probe_runtime
    tool_schemas.append(submit_schema or _submit_tool(submit_name, submit_description))

    result = run_agent(
        role_llm,
        max_steps=max_steps,
        system_prompt=instruction,
        initial_user_message=context,
        action_nudge=action_nudge,
        tool_schemas=tool_schemas,
        tool_handlers=tool_handlers,
        stop_when=tools.should_stop,
    )

    synthesis_steps = synthesis_peak = 0
    if tools.queries and not tools.submitted:
        synthesis_base = [
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
                    + tools.trace_path.read_text(errors="replace")
                ),
            },
        ]
        synthesis_messages = list(synthesis_base)
        synthesis_log = list(synthesis_base)
        last_error: str | None = None
        last_candidate: str | None = None
        final_validator = synthesis_validator or validator
        for _ in range(synthesis_attempts):
            reply = synthesis_llm.chat(synthesis_messages)
            synthesis_steps += 1
            synthesis_peak = max(synthesis_peak, reply.prompt_tokens)
            assistant_message = {"role": "assistant", "content": reply.content}
            synthesis_log.append(assistant_message)
            candidate = reply.content
            try:
                validated = final_validator(candidate)
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
                correction += " " + (
                    synthesis_instruction
                    or "Return only the complete artifact required by the system instruction."
                )
                correction_message = {"role": "user", "content": correction}
                synthesis_log.append(correction_message)
                # Retry from clean evidence. Keeping an invalid prose/code response in
                # context can make some providers repeat it verbatim indefinitely.
                synthesis_messages = [*synthesis_base, correction_message]
                continue
            atomic_write_text(output_path, validated)
            tools.save_submission(candidate)
            tools.submitted = True
            break
        _save_messages(f"{name}_synthesis", synthesis_log, workdir, artifact_dir)

    _save_role_transcript(name, result, workdir, artifact_dir)
    role_usage = role_llm.usage.as_dict()
    synthesis_usage = synthesis_llm.usage.as_dict()
    combined_usage = {
        key: role_usage[key] + synthesis_usage[key]
        for key in (
            "llm_calls",
            "prompt_tokens",
            "cache_hit_tokens",
            "completion_tokens",
        )
    }
    combined_usage["cost_yuan"] = round(
        role_usage["cost_yuan"] + synthesis_usage["cost_yuan"], 4
    )
    role = {
        "steps": result.steps + synthesis_steps,
        "errors": 0 if tools.submitted else 1,
        "format_errors": result.format_errors,
        "artifact_submitted": tools.submitted,
        "usage": combined_usage,
        "peak_ctx_tokens": max(result.peak_ctx_tokens, synthesis_peak),
        "tool_counts": result.tool_counts,
        "command_indexes": [],
        "submission_trace": tools.submission_trace,
        "runtime_probes": len(tools.probes),
        "runtime_probe_recommended": tools.runtime_probe_recommended,
        "runtime_probe_hint": tools.suggested_probe,
        "probe_trace": tools.probe_trace_path.name if tools.probes else None,
    }
    rag = {
        "dynamic": True,
        "trigger": trigger,
        "queries": tools.queries,
        "calls": len(tools.queries),
        "max_queries": max_queries,
        "usage": rag_llm.usage.as_dict(),
        "trace": tools.trace_path.name,
    }
    if not tools.queries:
        raise RuntimeError(f"{name} submitted no runtime-generated RAG query")
    if not tools.submitted:
        raise RoleSynthesisError(name, role, rag)
    return role, rag
