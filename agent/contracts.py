"""Task-context, code, and report/review contract plumbing.

Pure, dependency-light helpers shared by the orchestration pipeline: building the
public task context, validating generated code/reports/reviews, and invoking
optional workspace hooks. Kept separate from orchestration so the pipeline reads
as a state machine rather than a wall of string assembly.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Callable

from agent.generic_prompts import GENERIC_PROMPTS, RolePrompts
from agent.types import OracleConfig


def role_prompts() -> RolePrompts:
    return GENERIC_PROMPTS


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [b for b in blocks if "predictions.json" in b or "REPRO_RESULT" in b] or blocks
    return max(candidates, key=len).strip() + "\n"


def validate_report(content: str) -> str:
    content = content.strip()
    if len(content) < 300:
        raise ValueError("report must contain at least 300 characters")
    if "DSML" in content or "tool_calls" in content:
        raise ValueError("report contains tool-call markup instead of a synthesized artifact")
    return content + "\n"


def validate_review(content: str) -> str:
    content = validate_report(content)
    matches = re.findall(r"REVIEW_STATUS:\s*(PASS|REPAIR_REQUIRED)", content)
    if not matches:
        raise ValueError("review must end with REVIEW_STATUS: PASS or REPAIR_REQUIRED")
    body = re.sub(r"[*`]*REVIEW_STATUS:\s*(?:PASS|REPAIR_REQUIRED)[*`]*\s*$", "", content.rstrip()).rstrip()
    return f"{body}\n\nREVIEW_STATUS: {matches[-1]}\n"


def review_requires_repair(path: Path) -> bool:
    if not path.exists():
        return True
    return "REVIEW_STATUS: PASS" not in path.read_text(errors="replace")


def make_generic_code_validator(config: OracleConfig) -> Callable[[str], str]:
    artifact_markers = sorted(set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol)))
    if not config.public_result_protocol.strip():
        artifact_markers = ["REPRO_RESULT"]

    def validate(content: str) -> str:
        code = extract_python(content)
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


def generic_task_context(config: OracleConfig) -> str:
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


def call_workspace_hook(hook: Callable[..., None], workdir: Path) -> None:
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        hook()
        return
    if parameters:
        hook(workdir)
    else:
        hook()


__all__ = [
    "call_workspace_hook",
    "extract_python",
    "generic_task_context",
    "make_generic_code_validator",
    "review_requires_repair",
    "role_prompts",
    "validate_report",
    "validate_review",
]
