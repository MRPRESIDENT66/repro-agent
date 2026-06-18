"""Blind multi-agent reproduction pipeline.

Read this file top to bottom to see the whole system:

- ``PipelinePolicy``       — the three ablation conditions as data.
- ``provision_workspace``  — set up the blind sandbox + execution session.
- ``ReproductionPipeline`` — the role state machine (navigate -> reproduce ->
                             critique -> execute -> (review -> repair)*).
- ``build_run_record`` / ``emit_artifacts`` — serialize the run summary + outputs.
- ``run_oracle``           — thin driver: run the pipeline, verify, emit.

The agent never sees the hidden target; an independent verifier recomputes the
metric from per-sample artifacts. Each role starts from a fresh LLM context.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.contracts import (
    call_workspace_hook,
    generic_task_context,
    make_generic_code_validator,
    role_prompts,
    validate_report as default_validate_report,
    validate_review as default_validate_review,
)
from agent.diagnostics import make_generic_contract_diagnostics as _make_generic_contract_diagnostics
from agent.failure import classify_failure
from agent.llm import ChatLLM
from agent.repair import (
    apply_code_patch as _apply_code_patch,
    make_generic_repair_validator as _make_generic_repair_validator,
    patch_submission_adapter as _patch_submission_adapter,
    patch_tool as _patch_tool,
)
from agent.roles import (
    MAX_REPAIR_ROUNDS,
    RoleDeps,
    _clip,
    _dynamic_rag_role as _roles_dynamic_rag_role,
    _public_log,
    _require_handoff,
)
from agent.runtime_probe import MAX_RUNTIME_PROBES
from agent.types import OracleConfig
from retrieval.search import relevant_snippet, search_repo
from verify.check import verify_run


class _PipelineDone(Exception):
    """Clean early-stop for the solo ablation condition."""


@dataclass(frozen=True)
class PipelinePolicy:
    """The three ablation conditions, expressed as data instead of scattered flags.

    Collapsing ``run_critic`` / ``use_reviewer`` / ``post_mode`` / artifact-suffix
    into one object makes the conditions a single source of truth and trivially
    testable.
    """

    name: str
    run_critic: bool       # Navigator + Critic roles present
    use_reviewer: bool     # independent Reviewer between executions
    post_mode: str         # "none" (solo, one-shot) or "repair"
    artifact_suffix: str   # "" for full; the pipeline name otherwise

    @classmethod
    def from_name(cls, pipeline: str) -> "PipelinePolicy":
        table = {
            "solo": dict(run_critic=False, use_reviewer=False, post_mode="none"),
            "solo-repair": dict(run_critic=False, use_reviewer=False, post_mode="repair"),
            "full": dict(run_critic=True, use_reviewer=True, post_mode="repair"),
        }
        if pipeline not in table:
            raise ValueError(f"unknown pipeline {pipeline!r}; valid: {tuple(table)}")
        suffix = "" if pipeline == "full" else pipeline
        return cls(name=pipeline, artifact_suffix=suffix, **table[pipeline])


def provision_workspace(config: OracleConfig, workdir: Path, artifact_dir: Path) -> Any:
    """Set up the blind sandbox and return a fresh execution session.

    Copies clean source in, optionally asserts the workspace hides the target,
    clears generated leftovers, resets the artifact dir, and opens an (optionally
    network-isolated) execution session.
    """
    call_workspace_hook(config.copy_clean_source, workdir)
    if config.assert_blind_workspace is not None:
        call_workspace_hook(config.assert_blind_workspace, workdir)
    for pattern in ("*_probe_trace.md", "runtime_probes.json", "runtime_probes.sh"):
        for generated_path in workdir.glob(pattern):
            generated_path.unlink(missing_ok=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    session = config.make_session()
    if config.session_go_offline:
        session.go_offline()
    return session


def build_run_record(
    *,
    config: OracleConfig,
    pipeline: str,
    n_exec: int,
    roles: dict,
    rag: dict,
    workflow_error: str | None,
    rag_requirement: bool,
    handoff_requirement: bool,
    collaboration_pass: bool,
    public_evidence_found: bool,
    public_contract_diagnostics: list,
    verdict: Any,
    total_commands: int,
    probe_transcript: list,
    failure_classes: list,
) -> dict:
    """Assemble the serializable run summary (``result.json`` payload).

    Pure function of the run's observations — no I/O — so the report shape can be
    unit-tested without driving a full reproduction.
    """
    total_cost = round(
        sum(r["usage"].get("cost_yuan", 0.0) for r in roles.values())
        + sum(s["usage"].get("cost_yuan", 0.0) for s in rag.values()),
        4,
    )
    return {
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
        "public_evidence_found": public_evidence_found,
        "public_contract_diagnostics": public_contract_diagnostics,
        "verdict": verdict.as_dict(),
        "collaboration_pass": collaboration_pass,
        "total_cost_yuan": total_cost,
        "total_commands": total_commands,
        "runtime_probe_enabled": True,
        "runtime_probe_budget": MAX_RUNTIME_PROBES,
        "total_runtime_probes": len(probe_transcript),
        "failure_classes": failure_classes,
    }


def emit_artifacts(
    workdir: Path,
    artifact_dir: Path,
    result_json: str,
    session: Any,
    probe_transcript: list,
    *,
    handoff_files: tuple[str, ...],
    eval_script: str,
) -> None:
    """Serialize replay/probe scripts and mirror all run outputs to both dirs."""
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
        for handoff in handoff_files:
            src = workdir / handoff
            if src.exists() and output_dir != workdir:
                shutil.copy2(src, output_dir / handoff)
        src_eval = workdir / eval_script
        if src_eval.exists() and output_dir != workdir:
            shutil.copy2(src_eval, output_dir / eval_script)
        if output_dir != workdir:
            for trace in workdir.glob("*_rag_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)
            for trace in workdir.glob("*_probe_trace.md"):
                shutil.copy2(trace, output_dir / trace.name)


class ReproductionPipeline:
    """The role state machine for one blind reproduction attempt.

    Construction provisions the blind sandbox; :meth:`run` drives the stages per
    the :class:`PipelinePolicy`:

        navigate -> reproduce -> critique -> execute -> (review -> repair)*

    looping until the verifier-recomputable contract passes or the repair budget
    is spent. All run state lives on the instance so the orchestration reads as a
    short, linear ``run`` instead of a deep nest of closures.
    """

    def __init__(self, config: OracleConfig, policy: PipelinePolicy) -> None:
        self.config = config
        self.policy = policy
        self.prompts = role_prompts()
        self.task_context = generic_task_context(config)
        self.code_validator = make_generic_code_validator(config)
        self.validate_report = config.validate_report or default_validate_report
        self.validate_review = config.validate_review or default_validate_review
        self.contract_diagnostics = _make_generic_contract_diagnostics(config, pass_gate=self.passed)
        self.synthesis_instruction = (
            f"Return only the complete executable source code for {config.eval_script}. "
            "The program must produce the public result artifact when executed. "
            "Do not return the contents of predictions or result files."
        )

        self.workdir = config.workdir
        self.artifact_dir = config.artifact_dir
        if policy.artifact_suffix:
            self.artifact_dir = self.artifact_dir.parent / f"{self.artifact_dir.name}__{policy.artifact_suffix}"

        self.session = provision_workspace(config, self.workdir, self.artifact_dir)
        self.role_deps = RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        )

        self.roles: dict[str, dict] = {}
        self.rag: dict[str, dict] = {}
        self.workflow_error: str | None = None
        self.execution_start = 0
        self.latest_execution_start = 0
        self.n_exec = 0
        self.failure_classes: list[dict[str, str | None]] = []

    # --- shared helpers ---------------------------------------------------

    def passed(self, session: Any) -> bool:
        """Public pass gate: verifier-recomputable evidence exists and clears the
        random-chance floor. Never reads the hidden target."""
        config = self.config
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

    def rag_role(self, **kwargs: Any) -> tuple[dict, dict]:
        return _roles_dynamic_rag_role(
            task=self.config.task,
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
            **kwargs,
        )

    def _sync_eval_file(self) -> None:
        sync_file = getattr(self.session, "sync_file", None)
        if sync_file is not None and not sync_file(self.config.eval_script):
            raise RuntimeError(
                f"generated evaluation file is not visible to the execution session: {self.config.eval_script}"
            )

    # --- stages -----------------------------------------------------------

    def _navigate(self) -> None:
        self.roles["navigator"], self.rag["navigator"] = self.rag_role(
            name="navigator",
            instruction=self.prompts.navigator,
            context=self.task_context,
            output_path=self.workdir / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=self.validate_report,
            trigger="initial_task",
            max_steps=7,
        )

    def _reproduce(self) -> None:
        if self.policy.run_critic:
            builder_context = (
                "# Public task and result protocol\n\n"
                + self.task_context
                + "\n\n# Navigator handoff\n\n"
                + _require_handoff(self.workdir / "navigator_report.md", "navigator")
            )
        else:
            builder_context = self.task_context

        self.roles["reproducer"], self.rag["reproducer"] = self.rag_role(
            name="reproducer",
            instruction=self.prompts.reproducer,
            context=builder_context,
            output_path=self.workdir / self.config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete generated {self.config.eval_script}.",
            validator=self.code_validator,
            trigger="navigator_handoff" if self.policy.run_critic else "initial_task",
            max_steps=7,
            synthesis_instruction=self.synthesis_instruction,
            synthesis_attempts=5,
        )

    def _critique(self) -> None:
        critic_context = (
            "# Public task and result protocol\n\n"
            + self.task_context
            + "\n\n# Generated evaluation script\n\n"
            + (self.workdir / self.config.eval_script).read_text(errors="replace")
            + "\n\n# Navigator handoff\n\n"
            + _require_handoff(self.workdir / "navigator_report.md", "navigator")
        )
        self.roles["critic"], self.rag["critic"] = self.rag_role(
            name="critic",
            instruction=self.prompts.critic,
            context=critic_context,
            output_path=self.workdir / self.config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete audited {self.config.eval_script}.",
            validator=self.code_validator,
            trigger="generated_code_audit",
            max_steps=7,
            synthesis_instruction=self.synthesis_instruction,
            synthesis_attempts=5,
        )

    def _execute_reproducer(self) -> None:
        self._sync_eval_file()
        self.execution_start = len(self.session.transcript)
        eval_run = self.config.execute_eval(self.session)
        self.roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        self.roles["reproducer"]["command_indexes"] = [self.execution_start + 1, len(self.session.transcript)]
        self.session.write_file("reproducer_public_log.txt", _public_log(self.session, self.execution_start))
        self.latest_execution_start = self.execution_start

    def _review(self, round_index: int) -> None:
        diagnostics = self.contract_diagnostics(self.session)
        review_context = (
            "# Public task and result protocol\n\n"
            + self.task_context
            + "\n\n# Navigator handoff\n\n"
            + _require_handoff(self.workdir / "navigator_report.md", "navigator")
            + "\n\n# Evaluation implementation\n\n"
            + _clip((self.workdir / self.config.eval_script).read_text(errors="replace"), 12000)
            + "\n\n# Latest public execution log\n\n"
            + _clip(_public_log(self.session, self.latest_execution_start), 12000)
            + "\n\n# Deterministic public-contract audit\n\n"
            + "\n".join(f"- {issue}" for issue in diagnostics)
        )
        key = f"reviewer_{round_index}"
        self.roles[key], self.rag[key] = self.rag_role(
            name=key,
            instruction=self.prompts.reviewer,
            context=review_context,
            output_path=self.workdir / "review_report.md",
            submit_name="submit_review",
            submit_description="Submit the source-grounded execution audit.",
            validator=self.validate_review,
            trigger="execution_result" if round_index == 0 else "repair_execution_result",
            max_steps=6,
            max_queries=2,
        )

    def _repair_round(self, round_index: int) -> None:
        config = self.config
        diagnostics = self.contract_diagnostics(self.session)
        failure = classify_failure(session=self.session, diagnostics=diagnostics)
        self.failure_classes.append(
            {
                "round": str(round_index),
                "kind": failure.kind,
                "next_action": failure.next_action,
                "probe_hint": failure.probe_hint,
            }
        )
        parts = [
            "# Public task and result protocol\n\n" + self.task_context,
            "# Failure classification\n\n"
            f"- kind: {failure.kind}\n"
            f"- rationale: {failure.rationale}\n"
            f"- next_action: {failure.next_action}\n"
            + (f"- suggested_probe: {failure.probe_hint}\n" if failure.probe_hint else ""),
        ]
        parts.extend(
            [
                "# Current evaluation script\n\n" + (self.workdir / config.eval_script).read_text(errors="replace"),
                "# Latest public execution log\n\n" + _public_log(self.session, self.latest_execution_start),
            ]
        )
        if self.latest_execution_start != self.execution_start:
            parts.append("# Prior execution history (clipped)\n\n" + _clip(_public_log(self.session, self.execution_start), 6000))
        if self.policy.use_reviewer:
            parts.append("# Independent reviewer audit\n\n" + _require_handoff(self.workdir / "review_report.md", "reviewer"))
        if self.policy.run_critic:
            parts.append("# Navigator handoff\n\n" + _require_handoff(self.workdir / "navigator_report.md", "navigator"))
        parts.append("# Deterministic public-contract audit\n\n" + "\n".join(f"- {issue}" for issue in diagnostics))
        repair_context = "\n\n".join(parts)
        repair_validator = _make_generic_repair_validator(
            self.code_validator,
            self.session,
            self.workdir,
            self.execution_start,
            current_code=(self.workdir / config.eval_script).read_text(errors="replace"),
        )
        patch_validator = lambda payload, rv=repair_validator: _apply_code_patch(self.workdir / config.eval_script, payload, validate_code=rv)
        key = f"repair_{round_index}"
        self.roles[key], self.rag[key] = self.rag_role(
            name=key,
            instruction=self.prompts.repair.replace("{round_index}", str(round_index)),
            context=repair_context,
            output_path=self.workdir / config.eval_script,
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
            synthesis_instruction=self.synthesis_instruction
            + " The interactive patch phase did not submit a valid patch, so now return a complete repaired source file.",
            synthesis_validator=repair_validator,
            synthesis_attempts=4,
        )

        self._sync_eval_file()
        start = len(self.session.transcript)
        stepped_run = config.execute_eval(self.session)
        self.n_exec += 1
        self.latest_execution_start = start
        self.roles[key]["errors"] = 0 if stepped_run.ok else 1
        self.roles[key]["command_indexes"] = [start + 1, len(self.session.transcript)]
        self.session.write_file("reproducer_public_log.txt", _public_log(self.session, self.execution_start))

        if self.policy.use_reviewer:
            self._review(round_index)

    # --- driver -----------------------------------------------------------

    def run(self) -> "ReproductionPipeline":
        try:
            if self.policy.run_critic:
                self._navigate()
            self._reproduce()
            if self.policy.run_critic:
                self._critique()

            self._execute_reproducer()
            self.n_exec = 1
            if self.policy.post_mode == "none":
                raise _PipelineDone()

            if self.policy.use_reviewer:
                self._review(0)

            for round_index in range(1, MAX_REPAIR_ROUNDS + 1):
                if self.passed(self.session):
                    break
                self._repair_round(round_index)
        except _PipelineDone:
            pass
        except Exception as exc:
            self.workflow_error = f"{type(exc).__name__}: {exc}"
        finally:
            close = getattr(self.session, "close", None)
            if close is not None:
                close()
        return self


def run_oracle(config: OracleConfig, pipeline: str = "full") -> None:
    policy = PipelinePolicy.from_name(pipeline)
    pipe = ReproductionPipeline(config, policy).run()
    session = pipe.session

    verdict = verify_run(
        session.transcript,
        pipe.workdir,
        expected=config.expected,
        tolerance=config.tolerance,
        metric=config.metric,
        **config.verify_kwargs,
    )

    rag_requirement = bool(pipe.rag) and all(stage["dynamic"] and stage["calls"] >= 1 for stage in pipe.rag.values())
    handoff_requirement = True
    if policy.run_critic:
        handoff_requirement = (pipe.workdir / "navigator_report.md").exists()
    if policy.use_reviewer:
        handoff_requirement = handoff_requirement and (pipe.workdir / "review_report.md").exists()
    collaboration_pass = verdict.match and rag_requirement and handoff_requirement
    probe_transcript = list(getattr(session, "probe_transcript", []))

    record = build_run_record(
        config=config,
        pipeline=pipeline,
        n_exec=pipe.n_exec,
        roles=pipe.roles,
        rag=pipe.rag,
        workflow_error=pipe.workflow_error,
        rag_requirement=rag_requirement,
        handoff_requirement=handoff_requirement,
        collaboration_pass=collaboration_pass,
        public_evidence_found=pipe.passed(session),
        public_contract_diagnostics=pipe.contract_diagnostics(session),
        verdict=verdict,
        total_commands=len(session.transcript),
        probe_transcript=probe_transcript,
        failure_classes=pipe.failure_classes,
    )
    result_json = json.dumps(record, indent=2) + "\n"

    emit_artifacts(
        pipe.workdir,
        pipe.artifact_dir,
        result_json,
        session,
        probe_transcript,
        handoff_files=config.handoff_files,
        eval_script=config.eval_script,
    )

    print(result_json)


def _dynamic_rag_role(**kwargs: Any) -> tuple[dict, dict]:
    """Run a single dynamic RAG role outside a full pipeline.

    Defaults its dependencies from this module's ``ChatLLM`` / ``search_repo`` /
    ``relevant_snippet`` so targeted tests can monkeypatch them here.
    """
    deps = kwargs.pop(
        "deps",
        RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        ),
    )
    return _roles_dynamic_rag_role(**kwargs, deps=deps)


__all__ = [
    "PipelinePolicy",
    "ReproductionPipeline",
    "build_run_record",
    "emit_artifacts",
    "provision_workspace",
    "run_oracle",
]
