"""Top-level orchestration for blind multi-agent reproduction."""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from agent.diagnostics import (
    below_chance_diagnostic as _below_chance_diagnostic,
    latest_execution_observation as _latest_execution_observation,
    make_generic_contract_diagnostics as _make_generic_contract_diagnostics,
    workspace_artifact_snapshot as _workspace_artifact_snapshot,
)
from agent.failure import classify_failure
from agent.generic_prompts import GENERIC_PROMPTS, RolePrompts
from agent.llm import ChatLLM
from agent.repair import (
    apply_code_patch as _apply_code_patch,
    failed_import_packages as _failed_import_packages,
    make_generic_repair_validator as _make_generic_repair_validator,
    patch_submission_adapter as _patch_submission_adapter,
    patch_tool as _patch_tool,
)
from agent.roles import (
    MAX_REPAIR_ROUNDS,
    RoleDeps,
    _atomic_write_text,
    _clip,
    _dynamic_rag_role as _roles_dynamic_rag_role,
    _missing_path_hints,
    _public_log,
    _require_handoff,
    _search_evidence,
    _search_with_snippets as _roles_search_with_snippets,
    _submit_tool,
)
from agent.runtime_probe import (
    MAX_RUNTIME_PROBES,
    MAX_RUNTIME_PROBES_PER_ROLE,
    RUNTIME_PROBE_TOOL,
    runtime_probe_command as _runtime_probe_command,
    runtime_probe_observation as _runtime_probe_observation,
)
from agent.types import OracleConfig
from retrieval.search import relevant_snippet, search_repo
from verify.check import verify_run


class _PipelineDone(Exception):
    """Clean early-stop for the solo ablation condition."""


def _dynamic_rag_role(**kwargs: Any) -> tuple[dict, dict]:
    deps = kwargs.pop(
        "deps",
        RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        ),
    )
    return _roles_dynamic_rag_role(**kwargs, deps=deps)


def _role_prompts() -> RolePrompts:
    return GENERIC_PROMPTS


def _extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text.strip() + "\n"
    candidates = [b for b in blocks if "predictions.json" in b or "REPRO_RESULT" in b] or blocks
    return max(candidates, key=len).strip() + "\n"


def _validate_report(content: str) -> str:
    content = content.strip()
    if len(content) < 300:
        raise ValueError("report must contain at least 300 characters")
    if "DSML" in content or "tool_calls" in content:
        raise ValueError("report contains tool-call markup instead of a synthesized artifact")
    return content + "\n"


def _validate_review(content: str) -> str:
    content = _validate_report(content)
    matches = re.findall(r"REVIEW_STATUS:\s*(PASS|REPAIR_REQUIRED)", content)
    if not matches:
        raise ValueError("review must end with REVIEW_STATUS: PASS or REPAIR_REQUIRED")
    body = re.sub(r"[*`]*REVIEW_STATUS:\s*(?:PASS|REPAIR_REQUIRED)[*`]*\s*$", "", content.rstrip()).rstrip()
    return f"{body}\n\nREVIEW_STATUS: {matches[-1]}\n"


def _review_requires_repair(path: Path) -> bool:
    if not path.exists():
        return True
    return "REVIEW_STATUS: PASS" not in path.read_text(errors="replace")


def _search_with_snippets(
    query: str,
    llm: ChatLLM,
    workdir: Path,
    *,
    context: str | None = None,
    extra_exclude: set[str] | None = None,
    max_files: int = 4,
) -> str:
    deps = RoleDeps(
        llm_factory=ChatLLM,
        search_fn=search_repo,
        snippet_fn=relevant_snippet,
    )
    return _roles_search_with_snippets(
        query,
        llm,
        workdir,
        context=context,
        extra_exclude=extra_exclude,
        max_files=max_files,
        deps=deps,
    )


def _make_generic_code_validator(config: OracleConfig) -> Callable[[str], str]:
    artifact_markers = sorted(set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol)))
    if not config.public_result_protocol.strip():
        artifact_markers = ["REPRO_RESULT"]

    def validate(content: str) -> str:
        code = _extract_python(content)
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


def _generic_task_context(config: OracleConfig) -> str:
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


def _call_workspace_hook(hook: Callable[..., None], workdir: Path) -> None:
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        hook()
        return
    if parameters:
        hook(workdir)
    else:
        hook()


def run_oracle(config: OracleConfig, pipeline: str = "full") -> None:
    if pipeline not in ("solo", "solo-repair", "full"):
        raise ValueError(f"unknown pipeline {pipeline!r}; valid: ('solo', 'solo-repair', 'full')")
    prompts = _role_prompts()
    task_context = _generic_task_context(config)
    code_validator = _make_generic_code_validator(config)
    validate_report = config.validate_report or _validate_report
    validate_review = config.validate_review or _validate_review

    def _generic_pass_gate(session: Any) -> bool:
        recompute_fn = config.verify_kwargs.get("recompute_fn")
        if not callable(recompute_fn):
            return config.public_contract_passes(session)
        markers = sorted(set(re.findall(r"`([^`\n]+\.(?:json|jsonl|csv))`", config.public_result_protocol)))
        if markers and not all((config.workdir / m).is_file() for m in markers):
            return False
        try:
            probe = recompute_fn(config.workdir)
        except Exception:
            probe = None
        if not (isinstance(probe, tuple) and probe and isinstance(probe[0], (int, float))):
            return False
        if config.chance_level is not None and probe[0] < config.chance_level:
            return False
        return True

    generic_pass_gate = _generic_pass_gate
    contract_diagnostics = _make_generic_contract_diagnostics(config, pass_gate=generic_pass_gate)
    generic_code_synthesis_instruction = (
        f"Return only the complete executable source code for {config.eval_script}. "
        "The program must produce the public result artifact when executed. "
        "Do not return the contents of predictions or result files."
    )
    run_critic = pipeline == "full"
    post_mode = {"solo": "none", "solo-repair": "repair", "full": "repair"}[pipeline]
    use_reviewer = pipeline == "full"

    artifact_suffixes = []
    if pipeline != "full":
        artifact_suffixes.append(pipeline)
    workdir = config.workdir
    artifact_dir = config.artifact_dir
    if artifact_suffixes:
        artifact_dir = artifact_dir.parent / f"{artifact_dir.name}__{'__'.join(artifact_suffixes)}"

    _call_workspace_hook(config.copy_clean_source, workdir)
    if config.assert_blind_workspace is not None:
        _call_workspace_hook(config.assert_blind_workspace, workdir)
    for pattern in ("*_probe_trace.md", "runtime_probes.json", "runtime_probes.sh"):
        for generated_path in workdir.glob(pattern):
            generated_path.unlink(missing_ok=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    session = config.make_session()
    if config.session_go_offline:
        session.go_offline()

    role_deps = RoleDeps(
        llm_factory=ChatLLM,
        search_fn=search_repo,
        snippet_fn=relevant_snippet,
    )

    def rag_role(**kwargs: Any) -> tuple[dict, dict]:
        return _roles_dynamic_rag_role(
            task=config.task,
            workdir=workdir,
            artifact_dir=artifact_dir,
            session=session,
            search_extra_exclude=config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=role_deps,
            **kwargs,
        )

    roles: dict[str, dict] = {}
    rag: dict[str, dict] = {}
    workflow_error: str | None = None
    execution_start = 0
    n_exec = 0
    failure_classes: list[dict[str, str | None]] = []

    def sync_eval_file() -> None:
        sync_file = getattr(session, "sync_file", None)
        if sync_file is not None and not sync_file(config.eval_script):
            raise RuntimeError(
                f"generated evaluation file is not visible to the execution session: {config.eval_script}"
            )

    try:
        if run_critic:
            roles["navigator"], rag["navigator"] = rag_role(
                name="navigator",
                instruction=prompts.navigator,
                context=task_context,
                output_path=workdir / "navigator_report.md",
                submit_name="submit_handoff",
                submit_description="Submit the source-grounded Navigator handoff.",
                validator=validate_report,
                trigger="initial_task",
                max_steps=7,
            )
            builder_context = (
                "# Public task and result protocol\n\n"
                + task_context
                + "\n\n# Navigator handoff\n\n"
                + _require_handoff(workdir / "navigator_report.md", "navigator")
            )
        else:
            builder_context = task_context

        roles["reproducer"], rag["reproducer"] = rag_role(
            name="reproducer",
            instruction=prompts.reproducer,
            context=builder_context,
            output_path=workdir / config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete generated {config.eval_script}.",
            validator=code_validator,
            trigger="navigator_handoff" if run_critic else "initial_task",
            max_steps=7,
            synthesis_instruction=generic_code_synthesis_instruction,
            synthesis_attempts=5,
        )

        if run_critic:
            critic_context = (
                "# Public task and result protocol\n\n"
                + task_context
                + "\n\n# Generated evaluation script\n\n"
                + (workdir / config.eval_script).read_text(errors="replace")
                + "\n\n# Navigator handoff\n\n"
                + _require_handoff(workdir / "navigator_report.md", "navigator")
            )
            roles["critic"], rag["critic"] = rag_role(
                name="critic",
                instruction=prompts.critic,
                context=critic_context,
                output_path=workdir / config.eval_script,
                submit_name="submit_code",
                submit_description=f"Submit the complete audited {config.eval_script}.",
                validator=code_validator,
                trigger="generated_code_audit",
                max_steps=7,
                synthesis_instruction=generic_code_synthesis_instruction,
                synthesis_attempts=5,
            )

        sync_eval_file()
        execution_start = len(session.transcript)
        eval_run = config.execute_eval(session)
        roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        roles["reproducer"]["command_indexes"] = [execution_start + 1, len(session.transcript)]
        session.write_file("reproducer_public_log.txt", _public_log(session, execution_start))
        latest_execution_start = execution_start

        def review_current(round_index: int) -> None:
            diagnostics = contract_diagnostics(session)
            review_log_start = latest_execution_start
            review_context = (
                "# Public task and result protocol\n\n"
                + task_context
                + "\n\n# Navigator handoff\n\n"
                + _require_handoff(workdir / "navigator_report.md", "navigator")
                + "\n\n# Evaluation implementation\n\n"
                + _clip((workdir / config.eval_script).read_text(errors="replace"), 12000)
                + "\n\n# Latest public execution log\n\n"
                + _clip(_public_log(session, review_log_start), 12000)
                + "\n\n# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
            key = f"reviewer_{round_index}"
            roles[key], rag[key] = rag_role(
                name=key,
                instruction=prompts.reviewer,
                context=review_context,
                output_path=workdir / "review_report.md",
                submit_name="submit_review",
                submit_description="Submit the source-grounded execution audit.",
                validator=validate_review,
                trigger="execution_result" if round_index == 0 else "repair_execution_result",
                max_steps=6,
                max_queries=2,
            )

        n_exec = 1
        if post_mode == "none":
            raise _PipelineDone()

        if use_reviewer:
            review_current(0)

        for round_index in range(1, MAX_REPAIR_ROUNDS + 1):
            if generic_pass_gate(session):
                break
            diagnostics = contract_diagnostics(session)
            failure = classify_failure(session=session, diagnostics=diagnostics)
            failure_classes.append(
                {
                    "round": str(round_index),
                    "kind": failure.kind,
                    "next_action": failure.next_action,
                    "probe_hint": failure.probe_hint,
                }
            )
            parts = [
                "# Public task and result protocol\n\n" + task_context,
                "# Failure classification\n\n"
                f"- kind: {failure.kind}\n"
                f"- rationale: {failure.rationale}\n"
                f"- next_action: {failure.next_action}\n"
                + (f"- suggested_probe: {failure.probe_hint}\n" if failure.probe_hint else ""),
            ]
            parts.extend(
                [
                    "# Current evaluation script\n\n" + (workdir / config.eval_script).read_text(errors="replace"),
                    "# Latest public execution log\n\n" + _public_log(session, latest_execution_start),
                ]
            )
            if latest_execution_start != execution_start:
                parts.append("# Prior execution history (clipped)\n\n" + _clip(_public_log(session, execution_start), 6000))
            if use_reviewer:
                parts.append("# Independent reviewer audit\n\n" + _require_handoff(workdir / "review_report.md", "reviewer"))
            if run_critic:
                parts.append("# Navigator handoff\n\n" + _require_handoff(workdir / "navigator_report.md", "navigator"))
            parts.append("# Deterministic public-contract audit\n\n" + "\n".join(f"- {issue}" for issue in diagnostics))
            repair_context = "\n\n".join(parts)
            repair_validator = _make_generic_repair_validator(
                code_validator,
                session,
                workdir,
                execution_start,
                current_code=(workdir / config.eval_script).read_text(errors="replace"),
            )
            patch_validator = lambda payload, rv=repair_validator: _apply_code_patch(workdir / config.eval_script, payload, validate_code=rv)
            key = f"repair_{round_index}"
            roles[key], rag[key] = rag_role(
                name=key,
                instruction=prompts.repair.replace("{round_index}", str(round_index)),
                context=repair_context,
                output_path=workdir / config.eval_script,
                submit_name="submit_patch",
                submit_description=(
                    "Submit a small exact-replacement patch to the current eval script. "
                    "Use complete full-file replacement only if patch synthesis fails."
                ),
                validator=patch_validator,
                trigger="execution_error_and_reviewer_finding",
                max_steps=7,
                max_queries=3,
                submit_schema=_patch_tool("submit_patch", "Patch the current eval script with exact old/new replacements."),
                submission_adapter=_patch_submission_adapter,
                synthesis_instruction=generic_code_synthesis_instruction
                + " The interactive patch phase did not submit a valid patch, so now return a complete repaired source file.",
                synthesis_validator=repair_validator,
                synthesis_attempts=4,
            )

            sync_eval_file()
            start = len(session.transcript)
            stepped_run = config.execute_eval(session)
            n_exec += 1
            latest_execution_start = start
            roles[key]["errors"] = 0 if stepped_run.ok else 1
            roles[key]["command_indexes"] = [start + 1, len(session.transcript)]
            session.write_file("reproducer_public_log.txt", _public_log(session, execution_start))

            if use_reviewer:
                review_current(round_index)

    except _PipelineDone:
        pass
    except Exception as exc:
        workflow_error = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(session, "close", None)
        if close is not None:
            close()

    verdict = verify_run(
        session.transcript,
        workdir,
        expected=config.expected,
        tolerance=config.tolerance,
        metric=config.metric,
        **config.verify_kwargs,
    )

    rag_requirement = bool(rag) and all(stage["dynamic"] and stage["calls"] >= 1 for stage in rag.values())
    handoff_requirement = True
    if run_critic:
        handoff_requirement = (workdir / "navigator_report.md").exists()
    if use_reviewer:
        handoff_requirement = handoff_requirement and (workdir / "review_report.md").exists()
    collaboration_pass = verdict.match and rag_requirement and handoff_requirement
    total_cost = round(
        sum(r["usage"].get("cost_yuan", 0.0) for r in roles.values())
        + sum(s["usage"].get("cost_yuan", 0.0) for s in rag.values()),
        4,
    )
    probe_transcript = list(getattr(session, "probe_transcript", []))
    output = {
        "task": config.task,
        "pipeline": pipeline,
        "max_executions": MAX_REPAIR_ROUNDS + 1,
        "eval_executions": n_exec,
        "blind_workspace_checked": config.assert_blind_workspace is not None,
        "agents": len(roles),
        "attempt": config.attempt,
        "roles": roles,
        "rag": rag,
        "dynamic_rag": True,
        "retrieval_ranker": config.retrieval_ranker,
        "repair_mode": "patch_first_full_file_fallback",
        "workflow_error": workflow_error,
        "total_rag_calls": sum(stage["calls"] for stage in rag.values()),
        "rag_requirement_met": rag_requirement,
        "handoff_requirement_met": handoff_requirement,
        "public_evidence_found": generic_pass_gate(session),
        "public_contract_diagnostics": contract_diagnostics(session),
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": len(session.transcript),
        "runtime_probe_enabled": True,
        "runtime_probe_budget": MAX_RUNTIME_PROBES,
        "total_runtime_probes": len(probe_transcript),
        "failure_classes": failure_classes,
    }
    result_json = json.dumps(output, indent=2) + "\n"

    replay_fn = getattr(session, "replay_script", None)
    replay_script = (replay_fn() + "\n") if replay_fn is not None else None
    probe_replay_fn = getattr(session, "probe_replay_script", None)
    probe_replay_script = (
        (probe_replay_fn() + "\n") if probe_replay_fn is not None and probe_transcript else None
    )
    probe_json = json.dumps(
        [
            {
                "command": run.command,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "duration_s": run.duration_s,
            }
            for run in probe_transcript
        ],
        indent=2,
    ) + "\n"

    for output_dir in (workdir, artifact_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(result_json)
        if replay_script is not None:
            (output_dir / "commands.sh").write_text(replay_script)
        if probe_replay_script is not None:
            (output_dir / "runtime_probes.sh").write_text(probe_replay_script)
            (output_dir / "runtime_probes.json").write_text(probe_json)
        for handoff in config.handoff_files:
            src = workdir / handoff
            if src.exists() and output_dir != workdir:
                shutil.copy2(src, output_dir / handoff)
        src_eval = workdir / config.eval_script
        if src_eval.exists() and output_dir != workdir:
            shutil.copy2(src_eval, output_dir / config.eval_script)
        if output_dir != workdir:
            for trace in workdir.glob("*_rag_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)
            for trace in workdir.glob("*_probe_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)

    print(result_json)


__all__ = [
    "ChatLLM",
    "MAX_REPAIR_ROUNDS",
    "MAX_RUNTIME_PROBES",
    "MAX_RUNTIME_PROBES_PER_ROLE",
    "OracleConfig",
    "RUNTIME_PROBE_TOOL",
    "relevant_snippet",
    "run_oracle",
    "search_repo",
]
