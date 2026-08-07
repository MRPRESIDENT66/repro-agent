"""Task-context, code, and report/review validation.

Pure, dependency-light helpers shared by the orchestration pipeline: building the
public task context and validating generated code, reports, and reviews. Kept
separate so the pipeline reads as a state machine rather than string assembly.
"""

from __future__ import annotations

import ast
import re
from typing import Callable

from agent.types import OracleConfig


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    return max(blocks, key=len).strip() + "\n"


def public_artifact_names(protocol: str) -> list[str]:
    """Extract result filenames named in the public task contract."""
    return sorted(set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", protocol)))


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
    body = re.sub(
        r"[*`]*REVIEW_STATUS:\s*(?:PASS|REPAIR_REQUIRED)[*`]*\s*$",
        "",
        content.rstrip(),
    ).rstrip()
    if matches[-1] == "PASS":
        missing = []
        for category in ("model", "data", "preprocessing", "metric"):
            line = re.search(rf"(?im)^\s*-\s*`?{category}`?\s*:\s*(.+)$", body)
            if line is None or not re.search(
                r"`[^`\n]+\.(?:py|ya?ml|json|toml|cfg|ini|sh)(?::\d+)?`",
                line.group(1),
            ):
                missing.append(category)
        if missing:
            raise ValueError(
                "PASS review lacks source-path evidence for: " + ", ".join(missing)
            )
    return f"{body}\n\nREVIEW_STATUS: {matches[-1]}\n"


def make_generic_code_validator(config: OracleConfig) -> Callable[[str], str]:
    artifact_markers = public_artifact_names(config.public_result_protocol)

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
        f"The orchestrator will invoke the generated program as:\n"
        f"`{config.public_execution_command.strip()}`\n"
        "The program must accept and honor this command's arguments and "
        "provisioned paths.",
        "",
        "# Public result protocol",
        config.public_result_protocol.strip(),
        "",
        "The verifier accepts only this artifact contract. Generate it from",
        "the real evaluation; printed aggregate metrics are not evidence.",
    ]
    return "\n".join(lines)
