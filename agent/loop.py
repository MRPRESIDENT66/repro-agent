"""Sequential function-calling loop shared by all LLM roles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from agent.llm import LLM, Message, Reply


def _assistant_message(reply: Reply) -> Message:
    message: Message = {"role": "assistant", "content": reply.content}
    if reply.reasoning_content:
        message["reasoning_content"] = reply.reasoning_content
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in reply.tool_calls
        ]
    return message


def _tool_message(call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass
class AgentResult:
    steps: int
    completed: bool
    transcript: list[Message] = field(default_factory=list)
    peak_ctx_tokens: int = 0
    usage: dict = field(default_factory=dict)
    format_errors: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)


def run_agent(
    llm: LLM,
    *,
    system_prompt: str,
    initial_user_message: str,
    action_nudge: str,
    tool_schemas: list[dict],
    tool_handlers: dict[str, Callable[[dict], str]],
    stop_when: Callable[[], bool],
    max_steps: int = 12,
) -> AgentResult:
    """Run one role until its artifact contract is satisfied or budget expires."""
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message},
    ]
    peak_tokens = format_errors = 0
    tool_counts: dict[str, int] = {}
    usage_start = llm.usage.since(None) if hasattr(llm, "usage") else None
    steps_run = 0
    completed = False

    for step in range(1, max_steps + 1):
        steps_run = step
        reply = llm.chat(messages, tools=tool_schemas)
        peak_tokens = max(peak_tokens, reply.prompt_tokens)
        messages.append(_assistant_message(reply))

        if not reply.tool_calls:
            format_errors += 1
            messages.append({"role": "user", "content": action_nudge})
            continue

        call = reply.tool_calls[0]
        skipped = reply.tool_calls[1:]
        format_errors += len(skipped)
        tool_counts[call.name] = tool_counts.get(call.name, 0) + 1

        handler = tool_handlers.get(call.name)
        if handler is None:
            format_errors += 1
            observation = f"Unknown tool '{call.name}'."
        else:
            try:
                observation = handler(call.arguments)
            except Exception as exc:
                format_errors += 1
                observation = f"Tool failed: {exc}"
        messages.append(_tool_message(call.id, observation))

        for extra in skipped:
            messages.append(
                _tool_message(
                    extra.id,
                    "Skipped: exactly one tool call is executed per turn. "
                    "Call it again later if still needed.",
                )
            )
        if stop_when():
            completed = True
            break

    usage = llm.usage.since(usage_start).as_dict() if usage_start is not None else {}
    return AgentResult(
        steps=steps_run,
        completed=completed,
        transcript=messages,
        peak_ctx_tokens=peak_tokens,
        usage=usage,
        format_errors=format_errors,
        tool_counts=tool_counts,
    )
