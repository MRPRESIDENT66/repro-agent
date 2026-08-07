"""Blind multi-agent reproduction pipeline.

Read this file top to bottom to see the whole system:

- ``PipelinePolicy``       — the three ablation conditions as data.
- ``provision_workspace``  — set up the blind sandbox + execution session.
- ``ReproductionPipeline`` — the role state machine (navigate -> reproduce ->
                             critique -> execute -> (review -> repair)*).
- ``run_oracle``           — thin driver: run the pipeline, verify, emit.

The agent never sees the hidden target; an independent verifier recomputes the
metric from per-sample artifacts. Each role starts from a fresh LLM context.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent.artifacts import build_run_record, emit_artifacts
from agent.contracts import (
    generic_task_context,
    make_generic_code_validator,
    public_artifact_names,
    validate_report,
    validate_review,
)
from agent.diagnostics import make_generic_contract_diagnostics
from agent.failure import classify_failure
from agent.generic_prompts import GENERIC_PROMPTS
from agent.llm import ChatLLM
from agent.repair import (
    make_generic_repair_validator,
    make_patch_validator,
    patch_submission_adapter,
    patch_tool,
)
from agent.roles import (
    MAX_REPAIR_ROUNDS,
    RoleDeps,
    clip_text,
    public_log,
    require_handoff,
    run_rag_role,
)
from agent.types import OracleConfig
from retrieval.search import relevant_snippet, search_repo
from verify.check import verify_run


class _RunState(TypedDict):
    """Decision-relevant workflow state passed between LangGraph nodes.

    Large, auditable artifacts remain in ``workdir``.  State keeps their paths
    plus the execution metadata that determines the next route.
    """

    round: int
    eval_script_path: str
    navigator_report_path: str | None
    reviewer_report_path: str | None
    execution_start: int | None
    latest_execution_start: int | None
    n_exec: int
    last_execution_ok: bool | None
    failure_kind: str | None
    failure_next_action: str | None
    failure_probe_hint: str | None


@dataclass(frozen=True)
class PipelinePolicy:
    """One of the three supported ablation modes."""

    name: str

    @property
    def full_team(self) -> bool:
        return self.name == "full"

    @property
    def allow_repair(self) -> bool:
        return self.name != "solo"

    @property
    def artifact_suffix(self) -> str:
        return "" if self.name == "full" else self.name

    @classmethod
    def from_name(cls, pipeline: str) -> "PipelinePolicy":
        valid = ("solo", "solo-repair", "full")
        if pipeline not in valid:
            raise ValueError(f"unknown pipeline {pipeline!r}; valid: {valid}")
        return cls(name=pipeline)


def provision_workspace(config: OracleConfig, artifact_dir: Path) -> Any:
    """Set up the blind sandbox and return a fresh execution session.

    Copies clean source in, optionally asserts the workspace hides the target,
    clears generated leftovers, resets the artifact dir, and opens an (optionally
    network-isolated) execution session.
    """
    workdir = config.workdir
    config.copy_clean_source()
    if config.assert_blind_workspace is not None:
        config.assert_blind_workspace()
    for pattern in ("*_probe_trace.md", "runtime_probes.json", "runtime_probes.sh"):
        for generated_path in workdir.glob(pattern):
            generated_path.unlink(missing_ok=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    session = config.make_session()
    if config.session_go_offline:
        session.go_offline()
    return session


class ReproductionPipeline:
    """The role state machine for one blind reproduction attempt.

    Construction provisions the blind sandbox; :meth:`run` drives the stages per
    the :class:`PipelinePolicy`:

        navigate -> reproduce -> critique -> execute -> (review -> repair)*

    looping until the verifier-recomputable contract passes or the repair budget
    is spent. LangGraph State holds routing-relevant metadata while ``workdir``
    retains the executable and auditable artifacts.
    """

    def __init__(self, config: OracleConfig, policy: PipelinePolicy) -> None:
        self.config = config
        self.policy = policy
        self.prompts = GENERIC_PROMPTS
        self.task_context = generic_task_context(config)
        self.code_validator = make_generic_code_validator(config)
        self.contract_diagnostics = make_generic_contract_diagnostics(
            config, pass_gate=self.passed
        )
        self.synthesis_instruction = (
            f"Return only the complete executable source code for {config.eval_script}. "
            "The program must produce the public result artifact when executed. "
            "Do not return the contents of predictions or result files."
        )

        self.workdir = config.workdir
        self.artifact_dir = config.artifact_dir
        if policy.artifact_suffix:
            artifact_name = f"{self.artifact_dir.name}__{policy.artifact_suffix}"
            self.artifact_dir = self.artifact_dir.parent / artifact_name

        self.session = provision_workspace(config, self.artifact_dir)
        self.role_deps = RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        )

        self.roles: dict[str, dict] = {}
        self.rag: dict[str, dict] = {}
        self.workflow_error: str | None = None
        self.failure_classes: list[dict[str, str | None]] = []
        # Kept outside LangGraph State so an exception after execution (for
        # example, Reviewer synthesis failure) cannot erase the observed count.
        self.eval_executions_observed = 0
        self.run_state: _RunState = self._initial_state()

    def _initial_state(self) -> _RunState:
        return {
            "round": 0,
            "eval_script_path": self.config.eval_script,
            "navigator_report_path": None,
            "reviewer_report_path": None,
            "execution_start": None,
            "latest_execution_start": None,
            "n_exec": 0,
            "last_execution_ok": None,
            "failure_kind": None,
            "failure_next_action": None,
            "failure_probe_hint": None,
        }

    # --- shared helpers ---------------------------------------------------

    def passed(self, _session: Any) -> bool:
        """Public pass gate: verifier-recomputable evidence exists and clears the
        random-chance floor. Never reads the hidden target."""
        config = self.config
        markers = public_artifact_names(config.public_result_protocol)
        if markers and not all((config.workdir / m).is_file() for m in markers):
            return False
        try:
            probe = config.recompute_fn(config.workdir)
        except Exception:
            probe = None
        if not (isinstance(probe, tuple) and probe and isinstance(probe[0], (int, float))):
            return False
        if (
            config.expected_num_examples is not None
            and (len(probe) < 2 or probe[1] != config.expected_num_examples)
        ):
            return False
        if config.chance_level is not None and probe[0] < config.chance_level:
            return False
        return True

    def _sync_eval_file(self) -> None:
        if not self.session.sync_file(self.config.eval_script):
            raise RuntimeError(
                "generated evaluation file is not visible to the execution "
                f"session: {self.config.eval_script}"
            )

    # --- stages -----------------------------------------------------------

    def _node_navigate(self, _state: _RunState) -> dict[str, Any]:
        self.roles["navigator"], self.rag["navigator"] = run_rag_role(
            name="navigator",
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            instruction=self.prompts.navigator,
            context=self.task_context,
            output_path=self.workdir / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=validate_report,
            trigger="initial_task",
            max_steps=7,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )
        return {"navigator_report_path": "navigator_report.md"}

    def _node_reproduce(self, _state: _RunState) -> dict[str, Any]:
        if self.policy.full_team:
            builder_context = (
                "# Public task and result protocol\n\n"
                + self.task_context
                + "\n\n# Navigator handoff\n\n"
                + require_handoff(self.workdir / "navigator_report.md", "navigator")
            )
        else:
            builder_context = self.task_context

        self.roles["reproducer"], self.rag["reproducer"] = run_rag_role(
            name="reproducer",
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            instruction=self.prompts.reproducer,
            context=builder_context,
            output_path=self.workdir / self.config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete generated {self.config.eval_script}.",
            validator=self.code_validator,
            trigger="navigator_handoff" if self.policy.full_team else "initial_task",
            max_steps=7,
            synthesis_instruction=self.synthesis_instruction,
            synthesis_attempts=5,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )
        return {"eval_script_path": self.config.eval_script}

    def _node_critique(self, _state: _RunState) -> dict[str, Any]:
        critic_context = (
            "# Public task and result protocol\n\n"
            + self.task_context
            + "\n\n# Generated evaluation script\n\n"
            + (self.workdir / self.config.eval_script).read_text(errors="replace")
            + "\n\n# Navigator handoff\n\n"
            + require_handoff(self.workdir / "navigator_report.md", "navigator")
        )
        self.roles["critic"], self.rag["critic"] = run_rag_role(
            name="critic",
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
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
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )
        return {"eval_script_path": self.config.eval_script}

    def _node_execute(self, _state: _RunState) -> dict[str, Any]:
        self._sync_eval_file()
        execution_start = len(self.session.transcript)
        self.eval_executions_observed += 1
        eval_run = self.config.execute_eval(self.session)
        self.roles["reproducer"]["errors"] = 0 if eval_run.ok else 1
        self.roles["reproducer"]["command_indexes"] = [
            execution_start + 1,
            len(self.session.transcript),
        ]
        log = public_log(self.session, execution_start)
        self.session.write_file("reproducer_public_log.txt", log)
        update = {
            "execution_start": execution_start,
            "latest_execution_start": execution_start,
            "n_exec": 1,
            "last_execution_ok": eval_run.ok,
        }
        if self.policy.full_team:
            self._review(0, execution_start)
            update["reviewer_report_path"] = "review_report.md"
        return update

    def _review(self, round_index: int, latest_execution_start: int) -> None:
        diagnostics = self.contract_diagnostics(self.session)
        review_context = (
            "# Public task and result protocol\n\n"
            + self.task_context
            + "\n\n# Navigator handoff\n\n"
            + require_handoff(self.workdir / "navigator_report.md", "navigator")
            + "\n\n# Evaluation implementation\n\n"
            + clip_text(
                (self.workdir / self.config.eval_script).read_text(errors="replace"),
                12000,
            )
            + "\n\n# Latest public execution log\n\n"
            + clip_text(public_log(self.session, latest_execution_start), 12000)
            + "\n\n# Deterministic public-contract audit\n\n"
            + "\n".join(f"- {issue}" for issue in diagnostics)
        )
        key = f"reviewer_{round_index}"
        self.roles[key], self.rag[key] = run_rag_role(
            name=key,
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            instruction=self.prompts.reviewer,
            context=review_context,
            output_path=self.workdir / "review_report.md",
            submit_name="submit_review",
            submit_description="Submit the source-grounded execution audit.",
            validator=validate_review,
            trigger="execution_result" if round_index == 0 else "repair_execution_result",
            max_steps=6,
            max_queries=2,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )

    def _node_repair(self, state: _RunState) -> dict[str, Any]:
        round_index = state["round"] + 1
        config = self.config
        diagnostics = self.contract_diagnostics(self.session)
        failure = classify_failure(session=self.session, diagnostics=diagnostics)
        latest_execution_start = state["latest_execution_start"]
        execution_start = state["execution_start"]
        if latest_execution_start is None or execution_start is None:
            raise RuntimeError("repair requires a prior execution in LangGraph State")
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
                "# Current evaluation script\n\n"
                + (self.workdir / config.eval_script).read_text(errors="replace"),
                "# Latest public execution log\n\n"
                + public_log(self.session, latest_execution_start),
            ]
        )
        if latest_execution_start != execution_start:
            parts.append(
                "# Prior execution history (clipped)\n\n"
                + clip_text(public_log(self.session, execution_start), 6000)
            )
        if self.policy.full_team:
            parts.extend(
                [
                    "# Independent reviewer audit\n\n"
                    + require_handoff(self.workdir / "review_report.md", "reviewer"),
                    "# Navigator handoff\n\n"
                    + require_handoff(self.workdir / "navigator_report.md", "navigator"),
                ]
            )
        diagnostic_text = "\n".join(f"- {issue}" for issue in diagnostics)
        parts.append("# Deterministic public-contract audit\n\n" + diagnostic_text)
        repair_context = "\n\n".join(parts)
        repair_validator = make_generic_repair_validator(
            self.code_validator,
            self.session,
            self.workdir,
            execution_start,
            current_code=(self.workdir / config.eval_script).read_text(errors="replace"),
        )
        patch_validator = make_patch_validator(
            self.workdir / config.eval_script, repair_validator
        )
        key = f"repair_{round_index}"
        self.roles[key], self.rag[key] = run_rag_role(
            name=key,
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
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
            submit_schema=patch_tool(
                "submit_patch", "Patch the current eval script with exact old/new replacements."
            ),
            submission_adapter=patch_submission_adapter,
            synthesis_instruction=self.synthesis_instruction
            + " The interactive patch phase did not submit a valid patch, so now "
            "return a complete repaired source file.",
            synthesis_validator=repair_validator,
            synthesis_attempts=4,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )

        self._sync_eval_file()
        start = len(self.session.transcript)
        self.eval_executions_observed += 1
        stepped_run = config.execute_eval(self.session)
        self.roles[key]["errors"] = 0 if stepped_run.ok else 1
        self.roles[key]["command_indexes"] = [start + 1, len(self.session.transcript)]
        log = public_log(self.session, execution_start)
        self.session.write_file("reproducer_public_log.txt", log)

        if self.policy.full_team:
            self._review(round_index, start)

        return {
            "round": round_index,
            "latest_execution_start": start,
            "n_exec": state["n_exec"] + 1,
            "last_execution_ok": stepped_run.ok,
            "failure_kind": failure.kind,
            "failure_next_action": failure.next_action,
            "failure_probe_hint": failure.probe_hint,
            "reviewer_report_path": "review_report.md" if self.policy.full_team else None,
        }

    def _decide(self, state: _RunState) -> str:
        """Conditional edge: stop on a verifier pass, on the solo condition, or
        when the repair budget is spent; otherwise run another repair round."""
        if not self.policy.allow_repair:
            return "end"
        if self.passed(self.session):
            return "end"
        if state["round"] >= MAX_REPAIR_ROUNDS:
            return "end"
        return "repair"

    # --- driver -----------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(_RunState)
        graph.add_node("reproduce", self._node_reproduce)
        graph.add_node("execute", self._node_execute)
        graph.add_node("repair", self._node_repair)

        if self.policy.full_team:
            graph.add_node("navigate", self._node_navigate)
            graph.add_node("critique", self._node_critique)
            graph.set_entry_point("navigate")
            graph.add_edge("navigate", "reproduce")
            graph.add_edge("reproduce", "critique")
            graph.add_edge("critique", "execute")
        else:
            graph.set_entry_point("reproduce")
            graph.add_edge("reproduce", "execute")

        # navigate -> reproduce -> critique -> execute -> (repair)* -> END
        # _node_repair already runs its own execution, so after a repair we go
        # straight back to the decision rather than re-running execute.
        graph.add_conditional_edges("execute", self._decide, {"repair": "repair", "end": END})
        graph.add_conditional_edges("repair", self._decide, {"repair": "repair", "end": END})
        return graph.compile()

    def run(self) -> "ReproductionPipeline":
        try:
            self.run_state = self._build_graph().invoke(self.run_state)
        except Exception as exc:
            self.workflow_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.session.close()
        return self


def run_oracle(config: OracleConfig, pipeline: str = "full") -> None:
    policy = PipelinePolicy.from_name(pipeline)
    pipe = ReproductionPipeline(config, policy).run()
    session = pipe.session

    verdict = verify_run(
        pipe.workdir,
        expected=config.expected,
        tolerance=config.tolerance,
        metric=config.metric,
        expected_num_examples=config.expected_num_examples,
        recompute_fn=config.recompute_fn,
    )

    rag_requirement = bool(pipe.rag) and all(
        stage["dynamic"] and stage["calls"] >= 1 for stage in pipe.rag.values()
    )
    handoff_requirement = True
    if policy.full_team:
        handoff_requirement = all(
            (pipe.workdir / filename).exists()
            for filename in ("navigator_report.md", "review_report.md")
        )
    collaboration_pass = (
        verdict.match
        and pipe.workflow_error is None
        and rag_requirement
        and handoff_requirement
    )
    probe_transcript = list(session.probe_transcript)

    record = build_run_record(
        config=config,
        pipeline=pipeline,
        n_exec=pipe.eval_executions_observed,
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
        eval_script=config.eval_script,
    )

    print(result_json)
