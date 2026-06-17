"""Patch-first repair helpers for generated evaluation scripts."""

from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


def patch_tool(name: str, description: str, max_items: int = 8) -> dict:
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
                        "description": (
                            "Exact replacements to apply to the current file. "
                            "Each old string must be copied verbatim from the "
                            "current source and occur exactly once."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {
                                    "type": "string",
                                    "description": "Exact current source text to replace.",
                                },
                                "new": {
                                    "type": "string",
                                    "description": "Replacement source text.",
                                },
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


def patch_submission_adapter(arguments: dict) -> str:
    return json.dumps(arguments, ensure_ascii=False)


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


def apply_code_patch(
    current_path: Path,
    payload: str,
    *,
    validate_code: Callable[[str], str],
    required_change_terms: set[str] | None = None,
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
    if required_change_terms:
        changed = "\n".join(changed_fragments).lower()
        if not any(term.lower() in changed for term in required_change_terms):
            raise ValueError(
                f"patch does not address deterministic public-contract issue: "
                f"{sorted(required_change_terms)}"
            )
    return validate_code(updated)


def failed_import_packages(session: Any, workdir: Path, start: int = 0) -> set[str]:
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


def make_generic_repair_validator(
    base_validator: Callable[[str], str],
    session: Any,
    workdir: Path,
    execution_start: int,
    current_code: str | None = None,
) -> Callable[[str], str]:
    """Prevent a repair from re-entering package initializers proven to fail."""
    failed_packages = failed_import_packages(session, workdir, execution_start)

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


# Backward-compatible private names used by older tests/imports.
_patch_tool = patch_tool
_patch_submission_adapter = patch_submission_adapter
_apply_code_patch = apply_code_patch
_failed_import_packages = failed_import_packages
_make_generic_repair_validator = make_generic_repair_validator
