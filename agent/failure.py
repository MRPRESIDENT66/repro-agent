"""Rule-based failure classification for execution-driven Agent repair."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Failure:
    kind: str
    rationale: str
    next_action: str
    probe_hint: str | None = None


def _latest_text(session: Any) -> str:
    transcript = session.transcript
    if not transcript:
        return ""
    run = transcript[-1]
    return "\n".join((run.command, run.stdout, run.stderr))


def classify_failure(
    *,
    session: Any,
    diagnostics: list[str],
    workflow_error: str | None = None,
) -> Failure:
    text = "\n".join([workflow_error or "", _latest_text(session), "\n".join(diagnostics)])

    if workflow_error:
        missing_runtime_file = re.search(
            r"handoff missing|navigator_report|audit_report|"
            r"eval_.*\.py.*No such file",
            text,
        )
        if missing_runtime_file:
            return Failure(
                "workflow_state",
                "A required runtime handoff or generated eval file is missing.",
                "Recover runtime state or regenerate the missing artifact; do not "
                "treat this as a repository API problem.",
            )
        return Failure(
            "workflow_error",
            "The orchestrator failed outside the evaluated program.",
            "Fix runtime state first; only repair code after the generated eval "
            "script and handoffs exist.",
        )

    if session.transcript and session.transcript[-1].timed_out:
        return Failure(
            "timeout",
            "The evaluation exceeded the runtime budget before producing a "
            "complete result.",
            "Compare the executable work with the public task and Router audit "
            "requirements first. Remove unrequested algorithms, attacks, repeats, "
            "or dataset passes; then optimize batching/device/cache usage without "
            "reducing requested sample coverage or changing the metric.",
        )

    if re.search(r"SyntaxError|IndentationError|py_compile", text):
        return Failure(
            "syntax_error",
            "The generated program is syntactically invalid.",
            "Patch the smallest invalid code region named by the traceback.",
        )
    if re.search(r"ModuleNotFoundError|ImportError", text):
        module = re.search(r"No module named ['\"]([^'\"]+)", text)
        target = module.group(1) if module else None
        return Failure(
            "import_error",
            "The program imports a module unavailable in the runtime or triggers "
            "a broken import chain.",
            "Use repo search and an import_smoke probe; if the high-level API is "
            "unavailable, reuse source constants and semantics without that import "
            "chain.",
            f"import_smoke:{target}" if target else None,
        )
    if re.search(
        r"TypeError: .*unexpected keyword|TypeError: .*positional|missing .*required",
        text,
    ):
        return Failure(
            "api_mismatch",
            "The program called an API with the wrong signature.",
            "Inspect the exact source definition with python_signature or "
            "python_source, then patch only the call site.",
            "python_signature:<object named in traceback>",
        )
    if re.search(r"AttributeError: .*has no attribute", text):
        return Failure(
            "api_mismatch",
            "The program accessed an attribute unavailable in the installed API.",
            "Inspect the installed definition with python_source, then patch only "
            "the attribute access or call site.",
            "python_source:<object named in traceback>",
        )
    if re.search(r"FileNotFoundError|No such file or directory|Dataset not found", text):
        return Failure(
            "missing_path",
            "The program used a path or dataset layout that does not exist in the "
            "provisioned workspace.",
            "Use path_list probes and repository loader source; do not substitute "
            "a generic dataset layout.",
            "path_list:<nearest existing parent>",
        )
    if re.search(
        r"PicklingError|pickle error|can't get local object|cannot pickle|"
        r"can't pickle|attempting to start a worker process",
        text,
        re.I,
    ):
        return Failure(
            "multiprocessing_serialization",
            "A multiprocessing worker cannot serialize a local function, lambda, "
            "or another object captured by the data pipeline.",
            "Make dataset transforms and worker callables module-level and picklable, "
            "or set DataLoader num_workers=0 for the portable evaluation path. "
            "Patch only worker/callable construction; preserve data coverage and order.",
        )
    if re.search(
        r"missing after execution|missing: .*predictions|"
        r"public result artifact is missing",
        text,
        re.I,
    ):
        return Failure(
            "missing_artifact",
            "The evaluation ran without producing the required public result artifact.",
            "Patch the eval script to write the exact artifact path/schema from "
            "measured per-sample outputs in the working directory.",
        )
    if re.search(r"invalid JSON|malformed|wrong-count|exactly \d+", text, re.I):
        return Failure(
            "malformed_artifact",
            "The public artifact exists but does not match the required schema or count.",
            "Patch serialization/count/order while preserving measured predictions or scores.",
        )
    if re.search(r"outside_tolerance|below .*chance|semantically invalid|inverted", text, re.I):
        return Failure(
            "semantic_mismatch",
            "The verifier recomputed a metric from the artifact, but the measured "
            "semantics are wrong.",
            "Audit model/data/preprocessing/score direction/aggregation against "
            "repository evidence, then patch the semantic mismatch.",
        )
    return Failure(
        "unknown_failure",
        "The failure does not match a known class.",
        "Use the latest execution log and public contract diagnostics to identify the "
        "smallest repository-grounded repair.",
    )
