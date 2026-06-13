"""Run one strict-blind collaborative Multi-Agent + RAG RobustBench experiment.

Oracle: Carmon2019Unlabeled robust accuracy on CIFAR-10 (Linf, AutoAttack custom,
n=50, expected≈52.0 ± 2.0).

Same orchestration skeleton as run_openood_multi_rag.py — oracle-specific config
is confined to the PER-ORACLE CONTRACT block below.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from agent.llm import ChatLLM
from agent.loop import AgentResult, TOOLS, run_agent
from exec.session import Session
from retrieval.search import relevant_snippet, search_repo
from verify.check import extract_structured_evidence, verify_run

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# PER-ORACLE CONTRACT — all task-specific config here
# ---------------------------------------------------------------------------
SOURCE       = ROOT / "repos" / "robustbench"
MODEL_DIR    = ROOT / "repos" / "robustbench_models"
DATA_DIR     = ROOT / "repos" / "robustbench_data"
WORKDIR      = ROOT / "workspaces" / "robustbench_multi_rag"
ATTEMPT      = os.environ.get("ROBUSTBENCH_ATTEMPT", "001")
ARTIFACT_DIR = ROOT / "evals" / "runs" / f"robustbench_carmon_{ATTEMPT}"

EXPECTED     = 52.0
TOLERANCE    = 2.0
METRIC       = "robust_accuracy"

MODEL_NAME   = "Carmon2019Unlabeled"
DATASET      = "cifar10"
THREAT_MODEL = "Linf"
N_EXAMPLES   = 50
EPSILON      = 0.031372549
AA_ATTACKS   = ["apgd-ce", "apgd-dlr"]
AA_RESTARTS  = 1

TASK = f"""Reproduce the robust accuracy of {MODEL_NAME} on {DATASET} under the
{THREAT_MODEL} threat model using AutoAttack (custom version: {AA_ATTACKS},
{AA_RESTARTS} restart each).

The RobustBench repository (fixed commit), the pre-downloaded model checkpoint,
and the CIFAR-10 test data are already present on disk. The environment is
CPU-only and offline. Evaluate on the first {N_EXAMPLES} examples of the test
set with epsilon={EPSILON}."""

EVIDENCE = f"""A result counts only when an EXECUTED evaluation command prints:
REPRO_RESULT {{"metric":"{METRIC}","actual":<number>,"num_examples":{N_EXAMPLES}}}
The evaluation program itself must print the line. Do not echo or printf it.
`actual` is robust accuracy in percentage points (0–100)."""

SEARCH_REPO_TOOL = next(
    tool for tool in TOOLS if tool["function"]["name"] == "search_repo"
)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _copy_clean_source() -> None:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    shutil.copytree(
        SOURCE, WORKDIR,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    # Symlink pre-provisioned model and data (avoid re-download in offline env)
    (WORKDIR / "robustbench_models").symlink_to(MODEL_DIR)
    (WORKDIR / "robustbench_data").symlink_to(DATA_DIR)


def _clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:2200]}\n...[{len(text) - 4400} chars omitted]...\n{text[-2200:]}"


def _public_log(session: Session, start: int) -> str:
    parts = []
    for index, run in enumerate(session.transcript[start:], start + 1):
        parts.append(
            f"## Command {index}\n\n```bash\n{run.command}\n```\n\n"
            f"exit={run.exit_code} timed_out={run.timed_out}\n\n"
            f"```text\n{_clip(run.stdout)}\n{_clip(run.stderr)}\n```\n"
        )
    return "\n".join(parts)


def _save_role_transcript(name: str, result: AgentResult) -> None:
    text = "".join(json.dumps(message) + "\n" for message in result.transcript)
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_transcript.jsonl").write_text(text)


def _save_messages(name: str, messages: list[dict]) -> None:
    text = "".join(json.dumps(message) + "\n" for message in messages)
    for output_dir in (WORKDIR, ARTIFACT_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}_synthesis_transcript.jsonl").write_text(text)


def _combined_usage(*llms: ChatLLM) -> dict:
    usages = [llm.usage.as_dict() for llm in llms]
    return {
        "llm_calls": sum(u["llm_calls"] for u in usages),
        "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
        "cache_hit_tokens": sum(u["cache_hit_tokens"] for u in usages),
        "completion_tokens": sum(u["completion_tokens"] for u in usages),
        "cost_yuan": round(sum(u["cost_yuan"] for u in usages), 4),
    }


def _extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [b for b in blocks if "REPRO_RESULT" in b] or blocks
    return max(candidates, key=len).strip() + "\n"


def _submit_tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string",
                                            "description": "Complete file content"}},
                "required": ["content"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Code / report validation
# ---------------------------------------------------------------------------

def _validate_code(text: str) -> str:
    code = _extract_python(text)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"SyntaxError: {exc}") from exc
    for marker in ("REPRO_RESULT", "json.dumps"):
        if marker not in code:
            raise ValueError(f"Missing required marker: {marker!r}")
    return code


def _validate_report(text: str) -> str:
    if len(text.strip()) < 80:
        raise ValueError("Handoff too short")
    return text.strip()


def _validate_review(text: str) -> str:
    if "REVIEW_STATUS:" not in text:
        raise ValueError("Missing REVIEW_STATUS line")
    return text.strip()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _execute_eval(session: Session):
    """Run eval_robustbench.py with pre-provisioned model/data dirs."""
    # syntax check first
    session.shell("python -m py_compile eval_robustbench.py", timeout=30)
    return session.shell(
        f"PYTHONPATH=. python eval_robustbench.py "
        f"--model_name {MODEL_NAME} "
        f"--model_dir robustbench_models "
        f"--data_dir robustbench_data "
        f"--n_examples {N_EXAMPLES} "
        f"--epsilon {EPSILON}",
        timeout=900,  # AutoAttack on CPU can take ~10 min
    )


def _public_contract_passes(session: Session) -> bool:
    evidence = extract_structured_evidence(
        session.transcript,
        metric=METRIC,
        expected_num_examples=N_EXAMPLES,
    )
    return evidence is not None


def _public_contract_diagnostics(session: Session) -> list[str]:
    issues: list[str] = []
    evidence = extract_structured_evidence(
        session.transcript,
        metric=METRIC,
        expected_num_examples=N_EXAMPLES,
    )
    if evidence is None:
        issues.append(
            f"No valid REPRO_RESULT line found. Need "
            f'metric="{METRIC}" and num_examples={N_EXAMPLES} '
            f"in stdout of a successful command."
        )
    return issues


def _review_requires_repair(review_path: Path) -> bool:
    if not review_path.exists():
        return True
    return "REVIEW_STATUS: REPAIR_REQUIRED" in review_path.read_text()


def _repair_loop_should_continue(contract_passes: bool) -> bool:
    return not contract_passes


# ---------------------------------------------------------------------------
# RAG search
# ---------------------------------------------------------------------------

def _generated_files() -> set[str]:
    return {p.name for p in WORKDIR.glob("*_rag_trace.md")} | \
           {p.name for p in WORKDIR.glob("*_transcript.jsonl")} | \
           {p.name for p in WORKDIR.glob("*_synthesis_transcript.jsonl")}


def _search_with_snippets(query: str, llm: ChatLLM,
                           context: str | None = None) -> str:
    result = search_repo(query, WORKDIR, llm, k=5,
                         exclude_paths=_generated_files(),
                         context=context)
    lines = result.splitlines()
    parts = [result]
    for line in lines:
        path_str = line.strip().lstrip("  ").split("  —")[0]
        full = WORKDIR / path_str
        if full.exists() and full.suffix in {".py", ".md", ".txt", ".yaml", ".yml"}:
            snippet = relevant_snippet(full, query)
            parts.append(f"\n### {path_str}\n```\n{snippet}\n```")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dynamic RAG role — identical pattern to run_openood_multi_rag.py
# ---------------------------------------------------------------------------

def _dynamic_rag_role(
    *,
    name: str,
    session: Session,
    instruction: str,
    context: str,
    output_path: Path,
    submit_name: str,
    submit_description: str,
    validator: Callable[[str], str],
    trigger: str,
    max_steps: int = 7,
    max_queries: int = 3,
    submit_schema: dict | None = None,
    submission_adapter: Callable[[dict], str] | None = None,
    synthesis_instruction: str | None = None,
    synthesis_attempts: int = 3,
) -> tuple[dict, dict]:
    role_llm = ChatLLM()
    rag_llm = ChatLLM()
    synthesis_llm = ChatLLM()
    queries: list[str] = []
    submitted = False
    trace_sections = [
        f"# {name} dynamic RAG trace", "",
        f"Trigger: {trigger}", "",
        "Queries below were generated by the role at runtime.",
    ]
    trace_path = WORKDIR / f"{name}_rag_trace.md"
    submission_trace: str | None = None

    def save_trace() -> None:
        text = "\n".join(trace_sections) + "\n"
        for d in (WORKDIR, ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)
            (d / trace_path.name).write_text(text)

    def save_submission(raw: str) -> None:
        nonlocal submission_trace
        if submission_adapter is None:
            return
        submission_trace = f"{name}_submission.json"
        for d in (WORKDIR, ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)
            (d / submission_trace).write_text(raw.strip() + "\n")

    def dynamic_search(arguments: dict) -> str:
        query = str(arguments.get("query", "")).strip()
        if len(query) < 8:
            raise ValueError("query must describe the current uncertainty")
        if query in queries:
            raise ValueError("duplicate query; refine it from the latest evidence")
        if len(queries) >= max_queries:
            raise ValueError("dynamic RAG query budget exhausted; submit the artifact")
        result = _search_with_snippets(query, rag_llm, context=context)
        queries.append(query)
        trace_sections.extend([
            f"\n## Query {len(queries)}\n\n{query}",
            f"\n## Result {len(queries)}\n\n{result}",
        ])
        save_trace()
        return result

    def submit(arguments: dict) -> str:
        nonlocal submitted
        if not queries:
            raise ValueError("call search_repo with your own query before submitting")
        raw = (submission_adapter(arguments) if submission_adapter is not None
               else str(arguments.get("content", "")))
        content = validator(raw)
        output_path.write_text(content)
        save_submission(raw)
        submitted = True
        return f"accepted and wrote {output_path.name}"

    result = run_agent(
        TASK,
        session,
        role_llm,
        max_steps=max_steps,
        compress=False,
        use_tools=True,
        system_prompt=instruction,
        initial_user_message=context,
        action_nudge=(
            f"Call search_repo with a query derived from the current context, "
            f"or call {submit_name} when the artifact is grounded and complete."
        ),
        tool_schemas=[
            SEARCH_REPO_TOOL,
            submit_schema or _submit_tool(submit_name, submit_description),
        ],
        tool_handlers={
            "search_repo": dynamic_search,
            submit_name: submit,
        },
        stop_when=lambda: submitted or len(queries) >= max_queries,
        stop_summary=f"{name} search phase complete",
    )

    synthesis_steps = synthesis_peak = 0
    if queries and not submitted:
        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    instruction
                    + "\n\nThe dynamic repository search is complete. You have "
                    "no tools in this synthesis phase. "
                    + (synthesis_instruction or
                       "Return only the required complete artifact; do not "
                       "request or describe more searches.")
                ),
            },
            {
                "role": "user",
                "content": (
                    context
                    + "\n\n# Runtime-generated RAG trace\n\n"
                    + trace_path.read_text(errors="replace")
                ),
            },
        ]
        last_error: str | None = None
        last_candidate: str | None = None
        for _ in range(synthesis_attempts):
            reply = synthesis_llm.chat(synthesis_messages)
            synthesis_steps += 1
            synthesis_peak = max(synthesis_peak, reply.prompt_tokens)
            synthesis_messages.append({"role": "assistant", "content": reply.content})
            candidate = reply.content
            try:
                validated = validator(candidate)
            except Exception as exc:
                message = str(exc)
                near_identical = (
                    last_candidate is not None
                    and SequenceMatcher(None, last_candidate, candidate).ratio() > 0.97
                )
                repeated = message == last_error
                last_candidate, last_error = candidate, message
                correction = f"The synthesized artifact failed validation: {message}. Correct it."
                if near_identical:
                    correction += (
                        " Your artifact barely changed and still fails — change the "
                        "EXACT construct the error names."
                    )
                elif repeated:
                    correction += (
                        " This is the SAME error as your previous attempt — locate the "
                        "exact construct and change only that."
                    )
                synthesis_messages.append({"role": "user", "content": correction})
                continue
            output_path.write_text(validated)
            save_submission(candidate)
            submitted = True
            break
        _save_messages(f"{name}_synthesis", synthesis_messages)

    _save_role_transcript(name, result)

    if not queries:
        raise RuntimeError(f"{name} submitted no runtime-generated RAG query")
    if not submitted:
        raise RuntimeError(f"{name} failed to synthesize a valid artifact")

    role = {
        "steps": result.steps + synthesis_steps,
        "errors": result.errors,
        "format_errors": result.format_errors,
        "gave_final": submitted,
        "usage": _combined_usage(role_llm, synthesis_llm),
        "peak_ctx_tokens": max(result.peak_ctx_tokens, synthesis_peak),
        "tool_counts": result.tool_counts,
        "command_indexes": [],
        "submission_trace": submission_trace,
    }
    rag = {
        "dynamic": True,
        "trigger": trigger,
        "queries": queries,
        "calls": len(queries),
        "max_queries": max_queries,
        "usage": rag_llm.usage.as_dict(),
        "trace": trace_path.name,
    }
    return role, rag


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_multi_rag() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _copy_clean_source()

    session = Session(WORKDIR, venv_python=ROOT / ".venv" / "bin" / "python",
                      default_timeout=900)
    roles: dict = {}
    rag: dict = {}
    workflow_error: str | None = None

    try:
        # ── Navigator ─────────────────────────────────────────────────────
        roles["navigator"], rag["navigator"] = _dynamic_rag_role(
            name="navigator",
            session=session,
            instruction=f"""You are the Navigator in a collaborative ML reproduction team.
Search the RobustBench repository to understand:
- how to call load_model() to load {MODEL_NAME} from a pre-downloaded checkpoint
  (model_dir=robustbench_models; checkpoint is at robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt)
- how to load CIFAR-10 test data from data_dir=robustbench_data (first {N_EXAMPLES} examples)
  and what preprocessing is used
- how AutoAttack 'custom' version works: which attribute sets n_restarts, the
  exact API to run attacks {AA_ATTACKS}
- how robust accuracy is computed and returned (fraction or percentage?)
- CPU-only constraints

Write a concise grounded handoff with exact API calls and file paths.
Task: {TASK}""",
            context=TASK,
            output_path=WORKDIR / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=_validate_report,
            trigger="initial_task",
            max_steps=7,
        )

        builder_context = (
            "# Navigator handoff\n\n"
            + (WORKDIR / "navigator_report.md").read_text(errors="replace")
        )

        # ── Reproducer ────────────────────────────────────────────────────
        roles["reproducer"], rag["reproducer"] = _dynamic_rag_role(
            name="reproducer",
            session=session,
            instruction=f"""You are the Reproducer. Write a complete CPU-safe
`eval_robustbench.py` that:
- loads {MODEL_NAME} via robustbench's load_model() with model_dir=robustbench_models
- loads CIFAR-10 test data (first {N_EXAMPLES} examples) from data_dir=robustbench_data
  using the correct preprocessing from get_preprocessing()
- runs AutoAttack in 'custom' version with attacks_to_run={AA_ATTACKS},
  epsilon={EPSILON}, {AA_RESTARTS} restart per attack, on CPU
- computes robust accuracy as percentage points (0–100), NOT fraction
- accepts --model_name, --model_dir, --data_dir, --n_examples, --epsilon CLI args
- prints exactly one line: REPRO_RESULT {{"metric":"{METRIC}","actual":<pct>,"num_examples":{N_EXAMPLES}}}
  using json.dumps

Before coding, search for how to set n_restarts on the AutoAttack object.
Use robustbench imports directly; do not reimplement model loading or data loading.
{EVIDENCE}
Do not guess or mention the private target.""",
            context=builder_context,
            output_path=WORKDIR / "eval_robustbench.py",
            submit_name="submit_code",
            submit_description="Submit the complete eval_robustbench.py.",
            validator=_validate_code,
            trigger="navigator_handoff",
            max_steps=7,
            synthesis_attempts=5,
        )

        critic_context = (
            "# Generated script\n\n"
            + (WORKDIR / "eval_robustbench.py").read_text(errors="replace")
            + "\n\n# Navigator handoff\n\n"
            + (WORKDIR / "navigator_report.md").read_text(errors="replace")
        )

        # ── Critic ────────────────────────────────────────────────────────
        roles["critic"], rag["critic"] = _dynamic_rag_role(
            name="critic",
            session=session,
            instruction=f"""You are an independent Code Critic. Audit the generated
eval_robustbench.py against the RobustBench repository source. Verify:
- load_model() args: model_name, dataset, threat_model, model_dir
- get_preprocessing() is called correctly before load_clean_dataset()
- AutoAttack instantiation: norm='Linf', eps={EPSILON}, version='custom',
  attacks_to_run={AA_ATTACKS}, device=cpu
- n_restarts={AA_RESTARTS} set correctly (check exact attribute path in source)
- robust accuracy is percentage (0–100), not fraction (0–1)
- num_examples={N_EXAMPLES} in REPRO_RESULT
- CLI args accepted correctly

Submit a complete corrected script, not prose.
{EVIDENCE}
Do not guess or mention the private target.""",
            context=critic_context,
            output_path=WORKDIR / "eval_robustbench.py",
            submit_name="submit_code",
            submit_description="Submit the complete audited eval_robustbench.py.",
            validator=_validate_code,
            trigger="generated_code_audit",
            max_steps=7,
            synthesis_attempts=5,
        )

        execution_start = len(session.transcript)
        eval_run = _execute_eval(session)
        roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        roles["reproducer"]["command_indexes"] = [
            execution_start + 1, len(session.transcript)
        ]
        session.write_file(
            "reproducer_public_log.txt",
            _public_log(session, execution_start),
        )

        def review_current(round_index: int) -> None:
            diagnostics = _public_contract_diagnostics(session)
            review_context = (
                "# Navigator handoff\n\n"
                + (WORKDIR / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Evaluation implementation\n\n"
                + _clip((WORKDIR / "eval_robustbench.py").read_text(errors="replace"), 12000)
                + "\n\n# Public execution log\n\n"
                + _clip((WORKDIR / "reproducer_public_log.txt").read_text(errors="replace"), 12000)
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"reviewer_{round_index}"
            roles[key], rag[key] = _dynamic_rag_role(
                name=key,
                session=session,
                instruction="""You are the independent Reviewer. Audit the
implementation and execution log. Derive a search_repo query from the concrete
error or highest-risk semantic claim. The deterministic public-contract audit is
authoritative. When execution succeeded, check:
- robust accuracy is percentage (0–100), not fraction
- num_examples matches the expected count
- AutoAttack was actually run (not skipped)
- the result came from actual model evaluation
End with exactly `REVIEW_STATUS: PASS` or `REVIEW_STATUS: REPAIR_REQUIRED`.
Do not guess or mention the private target.""",
                context=review_context,
                output_path=WORKDIR / "review_report.md",
                submit_name="submit_review",
                submit_description="Submit the execution audit.",
                validator=_validate_review,
                trigger="execution_result" if round_index == 0 else "repair_execution_result",
                max_steps=6,
                max_queries=2,
            )

        review_current(0)

        for round_index in (1, 2, 3, 4):
            if not _repair_loop_should_continue(_public_contract_passes(session)):
                break
            diagnostics = _public_contract_diagnostics(session)
            repair_context = (
                "# Current script\n\n"
                + (WORKDIR / "eval_robustbench.py").read_text(errors="replace")
                + "\n\n# Public execution log\n\n"
                + _public_log(session, execution_start)
                + "\n\n# Reviewer audit\n\n"
                + (WORKDIR / "review_report.md").read_text(errors="replace")
                + "\n\n# Navigator handoff\n\n"
                + (WORKDIR / "navigator_report.md").read_text(errors="replace")
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"repair_{round_index}"
            roles[key], rag[key] = _dynamic_rag_role(
                name=key,
                session=session,
                instruction=f"""You are Repair Agent {round_index}. Fix the
concrete failure identified by the execution log and Reviewer. Search the repo
for the specific error or API question, then submit a corrected complete script.
Preserve: percentage accuracy, num_examples={N_EXAMPLES}, REPRO_RESULT format.
{EVIDENCE}
Do not guess or mention the private target.""",
                context=repair_context,
                output_path=WORKDIR / "eval_robustbench.py",
                submit_name="submit_code",
                submit_description="Submit the repaired eval_robustbench.py.",
                validator=_validate_code,
                trigger="execution_error_and_reviewer_finding",
                max_steps=7,
                max_queries=2,
                synthesis_attempts=4,
            )
            start = len(session.transcript)
            repaired_run = _execute_eval(session)
            roles[key]["errors"] = 0 if repaired_run.ok else 1
            roles[key]["command_indexes"] = [start + 1, len(session.transcript)]
            session.write_file(
                "reproducer_public_log.txt",
                _public_log(session, execution_start),
            )
            review_current(round_index)

    except Exception as exc:
        workflow_error = f"{type(exc).__name__}: {exc}"

    # ── Verify ────────────────────────────────────────────────────────────
    verdict = verify_run(
        session.transcript,
        workdir=WORKDIR,
        expected=EXPECTED,
        tolerance=TOLERANCE,
        metric=METRIC,
        expected_num_examples=N_EXAMPLES,
    )

    total_rag = sum(r.get("calls", 0) for r in rag.values())
    total_cost = sum(
        (roles[k].get("usage") or {}).get("cost_yuan", 0.0) for k in roles
    ) + sum(
        (rag[k].get("usage") or {}).get("cost_yuan", 0.0) for k in rag
    )

    result = {
        "task": TASK,
        "agents": len(roles),
        "attempt": ATTEMPT,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "workflow_error": workflow_error,
        "total_rag_calls": total_rag,
        "rag_requirement_met": total_rag >= 3,
        "public_evidence_found": _public_contract_passes(session),
        "public_contract_diagnostics": _public_contract_diagnostics(session),
        "verdict": verdict.as_dict(),
        "collaboration_pass": verdict.match,
        "total_cost_yuan": round(total_cost, 4),
        "total_commands": len(session.transcript),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_multi_rag()
