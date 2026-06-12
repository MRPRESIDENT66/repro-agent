"""The reproduction agent — a ReAct loop over shell actions.

Each turn the LLM sees the task + the transcript and emits **either** one bash
command (it writes scripts via heredoc) **or** ``FINAL: done``. We run the
command in the persistent :class:`~exec.session.Session`, feed back a truncated
observation, and repeat. On failure, repair escalates by consecutive error
(traceback → re-state environment → change approach) — the same tiered idea
carried over from the insight-agent self-repair.

M1 scope: the env is pre-provisioned (torch/datasets available); the agent's job
is to write+run the eval, recover from the real gotchas (dead links, API drift,
preprocessing), and report a number that deterministic verification can check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.llm import LLM, Message
from exec.session import Session

_BASH = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)
_SEARCH = re.compile(r"```search\s*\n(.*?)```", re.DOTALL)
_FINAL = re.compile(r"^\s*FINAL:\s*(.*?)\s*$", re.DOTALL)

SYSTEM = """You are an ML reproduction agent with a persistent bash shell in a \
fresh working directory. A Python env with torch, torchvision and the \
`datasets` library is already on PATH (use `python`). Files you write and \
anything you install persist across steps.

TASK: {task}
Reproduce the metric without access to the private published value.

Protocol — every reply is EITHER:
  1. exactly one ```bash code block: a single shell command. Write scripts with a
     heredoc, e.g.   cat > eval.py <<'EOF' ... EOF   then run them.
  2. a line `FINAL: done` only after an executed command printed a
     machine-readable result line (one per evaluated target):
     REPRO_RESULT {{"metric":"<metric_name>","actual":<number>,"num_examples":<int>}}
     For a multi-model task, also include "target":"<model_name>".
     The evaluation program itself must print this line; do not echo/printf it
     afterward. For percentage tasks, actual uses percentage points (91.06, not
     0.9106).
  3. if you have cloned a LARGE repo and need to find the eval entry/config, a
     ```search code block with a natural-language query (e.g. "evaluate resnet18
     on cifar10") — it returns the most relevant file paths in the repo.
You only see what commands print — print the metric clearly. Keep each step small.

STRATEGY:
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


def _compress(messages: list[Message], keep_recent: int = 4, max_old: int = 240) -> list[Message]:
    """Compress the long debug trajectory: keep system+task and the last few turns
    full, shrink older observations/tracebacks (stale, but the model still wants
    the gist). Bounds the context as the trajectory grows over many repair cycles.
    """
    if len(messages) <= 2 + keep_recent:
        return messages
    out = list(messages[:2])  # system + task, always full
    for m in messages[2:-keep_recent]:
        c = m["content"]
        if len(c) > max_old:
            c = c[:max_old] + f"\n[... {len(c) - max_old} chars compressed]"
        out.append({"role": m["role"], "content": c})
    out.extend(messages[-keep_recent:])
    return out


@dataclass
class AgentResult:
    final_raw: str | None          # the agent's FINAL text (raw)
    steps: int
    errors: int
    ran_eval: bool                 # at least one command produced stdout
    gave_final: bool
    transcript: list[Message] = field(default_factory=list)
    peak_ctx_chars: int = 0        # largest context actually sent to the LLM


def run_agent(task: str, session: Session, llm: LLM,
              max_steps: int = 12, compress: bool = False) -> AgentResult:
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM.format(task=task)},
        {"role": "user", "content": "Begin."},
    ]
    errors = consecutive = 0
    ran_eval = False
    peak = 0

    for step in range(1, max_steps + 1):
        view = _compress(messages) if compress else messages
        peak = max(peak, sum(len(m["content"]) for m in view))
        reply = llm.complete(view)
        messages.append({"role": "assistant", "content": reply})

        final = _FINAL.search(reply)
        if final:
            return AgentResult(final.group(1).strip(), step, errors, ran_eval, True, messages, peak)

        sq = _SEARCH.search(reply)
        if sq:
            from retrieval.search import search_repo
            obs = search_repo(sq.group(1).strip(), session.workdir, llm)
            messages.append({"role": "user", "content": f"Search results:\n{obs}"})
            continue

        m = _BASH.search(reply)
        if not m:
            messages.append({"role": "user", "content": "Reply with one ```bash block or `FINAL: done`."})
            continue

        r = session.shell(m.group(1).strip())
        if r.stdout.strip():
            ran_eval = True
        obs = _observe(r)
        if not r.ok:
            errors += 1
            consecutive += 1
            messages.append({"role": "user", "content": _repair(min(consecutive, 3), obs)})
        else:
            consecutive = 0
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    return AgentResult(None, max_steps, errors, ran_eval, False, messages, peak)
