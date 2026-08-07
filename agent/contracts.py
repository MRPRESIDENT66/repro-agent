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
    quoted = re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", protocol)
    unquoted = re.findall(
        r"(?<![\w/.-])([A-Za-z0-9_./-]+\.(?:json|jsonl|csv))\b", protocol
    )
    return sorted(set(quoted + unquoted))


def validate_report(content: str) -> str:
    content = content.strip()
    if len(content) < 300:
        raise ValueError("report must contain at least 300 characters")
    if "DSML" in content or "tool_calls" in content:
        raise ValueError("report contains tool-call markup instead of a synthesized artifact")
    return content + "\n"


def _audit_evidence_value(body: str, category: str) -> str | None:
    """Return one source-evidence value, ignoring harmless Markdown styling."""
    item = re.compile(
        r"^\s*-\s*(?:`(?P<code>[^`]+)`|\*\*(?P<bold>[^*]+)\*\*|"
        r"(?P<plain>[A-Za-z]+))\s*:?\s*(?P<value>.+)$"
    )
    for raw_line in body.splitlines():
        match = item.match(raw_line)
        if match is None:
            continue
        label = next(
            value
            for value in (match.group("code"), match.group("bold"), match.group("plain"))
            if value is not None
        )
        if label.rstrip(":").strip().lower() == category:
            return match.group("value")
    return None


def validate_audit(content: str) -> str:
    content = validate_report(content)
    matches = re.findall(r"AUDIT_STATUS:\s*(PASS|REPAIR_REQUIRED)", content)
    if not matches:
        raise ValueError("audit must end with AUDIT_STATUS: PASS or REPAIR_REQUIRED")
    body = re.sub(
        r"[*`]*AUDIT_STATUS:\s*(?:PASS|REPAIR_REQUIRED)[*`]*\s*$",
        "",
        content.rstrip(),
    ).rstrip()
    if matches[-1] == "PASS":
        missing = []
        for category in ("model", "data", "preprocessing", "metric"):
            value = _audit_evidence_value(body, category)
            if value is None or not re.search(
                r"`[^`\n]+\.(?:py|ya?ml|json|toml|cfg|ini|sh|md|rst|txt|ipynb)"
                r"(?::\d+(?:-\d+)?)?`",
                value,
            ):
                missing.append(category)
        if missing:
            raise ValueError(
                "PASS audit lacks source-path evidence for: " + ", ".join(missing)
            )
    return f"{body}\n\nAUDIT_STATUS: {matches[-1]}\n"


def make_generic_code_validator(config: OracleConfig) -> Callable[[str], str]:
    artifact_markers = public_artifact_names(config.public_result_protocol)
    literal_markers = [
        marker
        for marker in artifact_markers
        if marker not in config.public_execution_command
    ]

    def validate(content: str) -> str:
        code = extract_python(content)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"code is not syntactically valid: {exc}") from exc
        missing = [marker for marker in literal_markers if marker not in code]
        if missing:
            raise ValueError(
                "code does not produce the public result artifact described by the "
                f"runtime contract (missing: {missing})"
            )
        has_lambda = any(isinstance(node, ast.Lambda) for node in ast.walk(tree))
        has_worker_dataloader = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name != "DataLoader":
                continue
            worker_values = [
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "num_workers"
            ]
            if len(node.args) > 5:
                worker_values.append(node.args[5])
            for value in worker_values:
                try:
                    workers = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                if isinstance(workers, int) and workers > 0:
                    has_worker_dataloader = True
                    break
        if has_lambda and has_worker_dataloader:
            raise ValueError(
                "code combines a lambda with DataLoader num_workers > 0; spawn-based "
                "workers cannot pickle local lambdas. Use a module-level callable or "
                "set num_workers=0 without reducing sample coverage."
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
