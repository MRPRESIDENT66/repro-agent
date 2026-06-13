"""The reproduction agent — a ReAct loop over shell actions.

Each turn the LLM sees the task + the transcript and acts through **native
function calling** (tools: ``bash`` / ``search_repo`` / ``finish``). The legacy
text protocol — one ```bash code block or ``FINAL: done`` parsed by regex — is
kept behind ``use_tools=False`` as the ablation twin, so "tool calls vs text
parsing" is a measured comparison (format-error rate, success), not a fashion
choice. Both transports share the same evidence rules, repair tiers, and
compression.

We run each command in the persistent :class:`~exec.session.Session`, feed back
a truncated observation, and repeat. On failure, repair escalates by consecutive
error (traceback → re-state environment → change approach).

M1 scope: the env is pre-provisioned (torch/datasets available); the agent's job
is to write+run the eval, recover from the real gotchas (dead links, API drift,
preprocessing), and report a number that deterministic verification can check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from agent.llm import LLM, Message, Reply
from exec.session import Session

_BASH = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)
_SEARCH = re.compile(r"```search\s*\n(.*?)```", re.DOTALL)
_FINAL = re.compile(r"^\s*FINAL:\s*(.*?)\s*$", re.DOTALL)

_PREAMBLE = """You are an ML reproduction agent with a persistent bash shell in a \
fresh working directory. A Python env with torch, torchvision and the \
`datasets` library is already on PATH (use `python`). Files you write and \
anything you install persist across steps.

TASK: {task}
Reproduce the metric without access to the private published value."""

_PROTOCOL_TOOLS = """Protocol — every step, call exactly ONE tool:
  - bash(command): a single shell command. Write scripts with a heredoc, e.g.
    cat > eval.py <<'EOF' ... EOF   then run them.
  - search_repo(query): if you have cloned a LARGE repo and need to find the
    eval entry/config — a natural-language query (e.g. "evaluate resnet18 on
    cifar10") returns the most relevant file paths in the repo.
  - finish(summary): only after an executed command printed the REPRO_RESULT
    evidence line described below."""

_PROTOCOL_TEXT = """Protocol — every reply is EITHER:
  1. exactly one ```bash code block: a single shell command. Write scripts with a
     heredoc, e.g.   cat > eval.py <<'EOF' ... EOF   then run them.
  2. a line `FINAL: done` only after an executed command printed the REPRO_RESULT
     evidence line described below.
  3. if you have cloned a LARGE repo and need to find the eval entry/config, a
     ```search code block with a natural-language query (e.g. "evaluate resnet18
     on cifar10") — it returns the most relevant file paths in the repo."""

_EVIDENCE = """A result only counts when an EXECUTED command prints a machine-readable line
(one per evaluated target):
  REPRO_RESULT {"metric":"<metric_name>","actual":<number>,"num_examples":<int>}
For a multi-model task, also include "target":"<model_name>". The evaluation
program itself must print this line; do not echo/printf it afterward. For
percentage tasks, actual uses percentage points (91.06, not 0.9106).
You only see what commands print — print the metric clearly. Keep each step small."""

_STRATEGY = """STRATEGY:
  - For an UNFAMILIAR model, FIRST read its actual card/README PROSE (e.g. curl
    the raw README.md — not just API metadata) for the official load method.
  - If a loader reports the model is "not registered" / "unknown model", its
    architecture is almost certainly provided by a HELPER PACKAGE you must
    install and IMPORT (the card's usage section names it; importing it registers
    the model). Do that — do NOT rebuild the architecture by hand, which fails on
    state-dict key mismatches and burns your budget.
  - Once you know how to load it, RUN a full eval early to get a first number,
    then iterate on discrepancies. Don't seek the perfect setup before running.
  - If your number is close but off (e.g. 92.1 vs 92.6), the cause is usually
    preprocessing (normalization) — try standard alternatives.

Watch out (common and real):
  - dataset download links may be dead → use a mirror (e.g. HuggingFace `datasets`).
  - library APIs drift (dataset ids may need a namespace; function signatures change).
  - a model may need a specific load path (a registration import, a trust flag, a
    helper library named on its model card) — when unsure, read the card/README.
  - preprocessing matters (normalization, which label field) — read it from the
    model/config rather than guessing."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run one shell command in the persistent session. Write scripts "
                "with a heredoc (cat > eval.py <<'EOF' ... EOF), then run them. "
                "You only see what the command prints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "A single shell command."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": (
                "After cloning a LARGE repo: find the files most relevant to a "
                "natural-language goal (e.g. 'evaluate resnet18 on cifar10'). "
                "Returns ranked file paths from the repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Declare the task complete. Call ONLY after an executed command "
                "printed the REPRO_RESULT evidence line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-line summary."}
                },
                "required": [],
            },
        },
    },
]


def _system(task: str, use_tools: bool, evidence_instructions: str | None = None) -> str:
    protocol = _PROTOCOL_TOOLS if use_tools else _PROTOCOL_TEXT
    evidence = evidence_instructions or _EVIDENCE
    return "\n\n".join([_PREAMBLE.format(task=task), protocol, evidence, _STRATEGY])


def _truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.5), int(limit * 0.4)
    return f"{text[:head]}\n...[{len(text) - head - tail} chars truncated]...\n{text[-tail:]}"


def _observe(r) -> str:
    head = (
        f"[timed out after {r.duration_s:.0f}s]"
        if r.timed_out
        else f"[exit {r.exit_code} in {r.duration_s:.0f}s]"
    )
    parts = [head]
    if r.stdout.strip():
        parts.append("stdout:\n" + _truncate(r.stdout))
    if r.stderr.strip():
        parts.append("stderr:\n" + _truncate(r.stderr))
    return "\n".join(parts) if len(parts) > 1 else head + " (no output)"


def _repair(tier: int, obs: str) -> str:
    base = f"Observation:\n{obs}\n\n"
    if tier <= 1:
        return base + "That failed. Read the error and fix it."
    if tier == 2:
        return base + (
            "Still failing. Recall: torch/torchvision/datasets are installed and "
            "on PATH; dataset links may be dead (use a HF mirror); check API "
            "signatures. Re-examine and fix."
        )
    return base + "Still failing after retries. Abandon this approach and try a different method."


def _msg_len(m: Message) -> int:
    n = len(m["content"]) if isinstance(m.get("content"), str) else 0
    for tc in m.get("tool_calls", ()):  # in FC mode the command text lives here
        n += len(tc["function"]["arguments"])
    return n


def _assistant_msg(reply: Reply) -> Message:
    msg: Message = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)},
            }
            for c in reply.tool_calls
        ]
    return msg


def _tool_msg(call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _compress(messages: list[Message], keep_recent: int = 4, max_old: int = 240) -> list[Message]:
    """Compress the long debug trajectory: keep system+task and the last few turns
    full, shrink older observations/tracebacks (stale, but the model still wants
    the gist). Bounds the context as the trajectory grows over many repair cycles.

    Structure-preserving: no message is dropped and non-content fields
    (``tool_calls`` / ``tool_call_id``) pass through untouched, so the
    assistant-call ↔ tool-result pairing the API requires stays valid.
    """
    if len(messages) <= 2 + keep_recent:
        return messages
    out = list(messages[:2])  # system + task, always full
    for m in messages[2:-keep_recent]:
        c = m.get("content")
        if isinstance(c, str) and len(c) > max_old:
            m = {**m, "content": c[:max_old] + f"\n[... {len(c) - max_old} chars compressed]"}
        out.append(m)
    out.extend(messages[-keep_recent:])
    return out


@dataclass
class AgentResult:
    final_raw: str | None          # the agent's FINAL/finish text (raw)
    steps: int
    errors: int                    # commands that exited non-zero (shell errors)
    ran_eval: bool                 # at least one command produced stdout
    gave_final: bool
    transcript: list[Message] = field(default_factory=list)
    peak_ctx_chars: int = 0        # largest context actually sent to the LLM
    peak_ctx_tokens: int = 0       # same, in real tokens (max prompt_tokens/call)
    usage: dict = field(default_factory=dict)  # this run's tokens + yuan (delta)
    format_errors: int = 0         # turns the model violated the action protocol
                                   # (text: unparseable reply; FC: no/empty/bad tool call)
                                   # — the metric the FC-vs-text ablation turns on
    tool_counts: dict[str, int] = field(default_factory=dict)


def run_agent(task: str, session: Session, llm: LLM,
              max_steps: int = 12, compress: bool = False,
              use_tools: bool = True,
              evidence_instructions: str | None = None,
              system_prompt: str | None = None,
              initial_user_message: str = "Begin.",
              action_nudge: str | None = None,
              tool_schemas: list[dict] | None = None,
              tool_handlers: dict[str, Callable[[dict], str]] | None = None,
              stop_when: Callable[[], bool] | None = None,
              stop_summary: str = "stage contract satisfied") -> AgentResult:
    messages: list[Message] = [
        {
            "role": "system",
            "content": system_prompt or _system(task, use_tools, evidence_instructions),
        },
        {"role": "user", "content": initial_user_message},
    ]
    errors = consecutive = format_errors = 0
    ran_eval = False
    peak_chars = peak_tokens = 0
    tool_counts: dict[str, int] = {}
    usage_start = llm.usage.since(None) if hasattr(llm, "usage") else None
    active_tools = TOOLS if tool_schemas is None else tool_schemas
    custom_handlers = tool_handlers or {}

    def done(final_raw: str | None, steps: int, gave_final: bool) -> AgentResult:
        usage = llm.usage.since(usage_start).as_dict() if usage_start is not None else {}
        return AgentResult(final_raw, steps, errors, ran_eval, gave_final, messages,
                           peak_ctx_chars=peak_chars, peak_ctx_tokens=peak_tokens,
                           usage=usage, format_errors=format_errors,
                           tool_counts=dict(tool_counts))

    def run_bash(command: str) -> str:
        """Execute + classify into observation-or-repair text (shared by both modes)."""
        nonlocal errors, consecutive, ran_eval
        r = session.shell(command)
        if r.stdout.strip():
            ran_eval = True
        obs = _observe(r)
        if not r.ok:
            errors += 1
            consecutive += 1
            return _repair(min(consecutive, 3), obs)
        consecutive = 0
        return f"Observation:\n{obs}"

    for step in range(1, max_steps + 1):
        view = _compress(messages) if compress else messages
        peak_chars = max(peak_chars, sum(_msg_len(m) for m in view))
        reply = llm.chat(view, tools=active_tools if use_tools else None)
        peak_tokens = max(peak_tokens, reply.prompt_tokens)
        messages.append(_assistant_msg(reply))

        if use_tools:
            if not reply.tool_calls:
                final = _FINAL.search(reply.content or "")
                if final:
                    return done(final.group(1).strip(), step, True)
                format_errors += 1  # prose instead of a tool call
                messages.append({
                    "role": "user",
                    "content": action_nudge or (
                        "Call the bash tool with one command, or finish "
                        "once the evidence line was printed."
                    ),
                })
                continue
            # The protocol promises one action per step. Providers may still
            # return several parallel tool calls; execute only the first and
            # acknowledge the rest so assistant↔tool pairing remains valid.
            call, *skipped = reply.tool_calls
            if skipped:
                format_errors += len(skipped)

            tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
            if call.name in custom_handlers:
                try:
                    observation = custom_handlers[call.name](call.arguments)
                except Exception as exc:
                    format_errors += 1
                    observation = f"Custom tool failed: {exc}"
                messages.append(_tool_msg(call.id, observation))
            elif call.name == "bash":
                command = str(call.arguments.get("command", "")).strip()
                if not command:
                    format_errors += 1  # empty/malformed bash arguments
                    messages.append(_tool_msg(call.id, "bash requires a 'command' string argument."))
                else:
                    messages.append(_tool_msg(call.id, run_bash(command)))
            elif call.name == "search_repo":
                from retrieval.search import search_repo
                obs = search_repo(str(call.arguments.get("query", "")), session.workdir, llm)
                messages.append(_tool_msg(call.id, f"Search results:\n{obs}"))
            elif call.name == "finish":
                final_summary = str(call.arguments.get("summary") or "done")
                messages.append(_tool_msg(call.id, "finished."))
                for extra in skipped:
                    messages.append(_tool_msg(
                        extra.id,
                        "Skipped: exactly one tool call is executed per turn.",
                    ))
                return done(final_summary, step, True)
            else:
                format_errors += 1  # hallucinated tool name
                messages.append(_tool_msg(call.id, f"Unknown tool '{call.name}'."))

            for extra in skipped:
                messages.append(_tool_msg(
                    extra.id,
                    "Skipped: exactly one tool call is executed per turn. "
                    "Call this tool again on a later turn if still needed.",
                ))
            if stop_when is not None and stop_when():
                return done(stop_summary, step, True)
            continue

        # --- legacy text protocol (the ablation twin) ---
        text = reply.content or ""
        final = _FINAL.search(text)
        if final:
            return done(final.group(1).strip(), step, True)

        sq = _SEARCH.search(text)
        if sq:
            from retrieval.search import search_repo
            obs = search_repo(sq.group(1).strip(), session.workdir, llm)
            messages.append({"role": "user", "content": f"Search results:\n{obs}"})
            continue

        m = _BASH.search(text)
        if not m:
            format_errors += 1  # reply had no parseable ```bash / search / FINAL
            messages.append({"role": "user", "content": "Reply with one ```bash block or `FINAL: done`."})
            continue
        messages.append({"role": "user", "content": run_bash(m.group(1).strip())})

    return done(None, max_steps, False)
