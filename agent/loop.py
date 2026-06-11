"""The reproduction agent — a ReAct loop over shell actions.

Each turn the LLM sees the task + the transcript and emits **either** one bash
command (it writes scripts via heredoc) **or** ``FINAL: <number>``. We run the
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
_FINAL = re.compile(r"FINAL:\s*(.*)", re.DOTALL)

SYSTEM = """You are an ML reproduction agent with a persistent bash shell in a \
fresh working directory. A Python env with torch, torchvision and the \
`datasets` library is already on PATH (use `python`). Files you write and \
anything you install persist across steps.

TASK: {task}
The repo's published value is {expected}. Reproduce it.

Protocol — every reply is EITHER:
  1. exactly one ```bash code block: a single shell command. Write scripts with a
     heredoc, e.g.   cat > eval.py <<'EOF' ... EOF   then run them.
  2. a line `FINAL: <number>` once you have the reproduced metric.
You only see what commands print — print the metric clearly. Keep each step small.

STRATEGY — act, don't over-analyze:
  - Do NOT try to find the perfect setup before running. Write a complete eval
    script and RUN it early to get a first number, then iterate on discrepancies.
  - If your number is close but doesn't match (e.g. you get 92.1 vs 92.6), the
    cause is usually preprocessing — try standard alternatives (e.g. a different
    CIFAR normalization) rather than reading source for hours.
  - You have a limited step budget; spend it running, not browsing files.

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


@dataclass
class AgentResult:
    final_raw: str | None          # the agent's FINAL text (raw)
    steps: int
    errors: int
    ran_eval: bool                 # at least one command produced stdout
    gave_final: bool
    transcript: list[Message] = field(default_factory=list)


def run_agent(task: str, expected: float, session: Session, llm: LLM, max_steps: int = 12) -> AgentResult:
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM.format(task=task, expected=expected)},
        {"role": "user", "content": "Begin."},
    ]
    errors = consecutive = 0
    ran_eval = False

    for step in range(1, max_steps + 1):
        reply = llm.complete(messages)
        messages.append({"role": "assistant", "content": reply})

        final = _FINAL.search(reply)
        if final:
            return AgentResult(final.group(1).strip(), step, errors, ran_eval, True, messages)

        m = _BASH.search(reply)
        if not m:
            messages.append({"role": "user", "content": "Reply with one ```bash block or `FINAL: <number>`."})
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

    return AgentResult(None, max_steps, errors, ran_eval, False, messages)
