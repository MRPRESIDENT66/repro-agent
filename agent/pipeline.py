"""Blind multi-agent reproduction pipeline.

Read this file top to bottom to see the whole system:

- ``PipelinePolicy``       — the four ablation conditions as data.
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
from agent.router import RouteDecision, route_task
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
    public_contract_ok: bool | None
    audit_complete: bool
    audit_requires_repair: bool
    repeated_failure: bool


@dataclass(frozen=True)
class PipelinePolicy:
    """One of the four supported ablation modes."""

    name: str

    @property
    def full_team(self) -> bool:
        return self.name == "full"

    @property
    def adaptive(self) -> bool:
        return self.name == "adaptive"

    @property
    def allow_repair(self) -> bool:
        return self.name != "solo"

    @property
    def artifact_suffix(self) -> str:
        return "" if self.name == "full" else self.name

    @classmethod
    def from_name(cls, pipeline: str) -> "PipelinePolicy":
        valid = ("solo", "solo-repair", "full", "adaptive")
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

    looping until the public artifact contract passes or the repair budget is
    spent. The private verifier runs once after this workflow. LangGraph State
    holds routing metadata while ``workdir`` retains auditable artifacts.
    """

    def __init__(self, config: OracleConfig, policy: PipelinePolicy) -> None:
        self.config = config
        self.policy = policy
        self.prompts = GENERIC_PROMPTS
        self.task_context = generic_task_context(config)
        self.code_validator = make_generic_code_validator(config)
        self.contract_diagnostics = make_generic_contract_diagnostics(config)
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
        self.route_decision: RouteDecision | None = (
            route_task(config, self.workdir) if policy.adaptive else None
        )
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
            "public_contract_ok": None,
            "audit_complete": False,
            "audit_requires_repair": False,
            "repeated_failure": False,
        }

    # --- shared helpers ---------------------------------------------------

    def passed(self, _session: Any) -> bool:
        """Return whether the artifact satisfies the public schema only."""
        try:
            return bool(self.config.public_check_fn(self.config.workdir))
        except Exception:
            return False

    def _sync_eval_file(self) -> None:
        if not self.session.sync_file(self.config.eval_script):
            raise RuntimeError(
                "generated evaluation file is not visible to the execution "
                f"session: {self.config.eval_script}"
            )

    def _clear_public_artifacts(self) -> None:
        for name in public_artifact_names(self.config.public_result_protocol):
            path = self.workdir / name
            if path.is_file():
                path.unlink()

    def _execution_update(
        self, state: _RunState, *, start: int, ok: bool, n_exec: int
    ) -> dict[str, Any]:
        public_ok = self.passed(self.session)
        failure = None
        repeated = False
        if not public_ok:
            failure = classify_failure(
                session=self.session,
                diagnostics=self.contract_diagnostics(self.session),
            )
            repeated = bool(
                self.failure_classes
                and self.failure_classes[-1].get("kind") == failure.kind
            )
        return {
            "latest_execution_start": start,
            "n_exec": n_exec,
            "last_execution_ok": ok,
            "failure_kind": failure.kind if failure else state["failure_kind"],
            "failure_next_action": (
                failure.next_action if failure else state["failure_next_action"]
            ),
            "failure_probe_hint": (
                failure.probe_hint if failure else state["failure_probe_hint"]
            ),
            "public_contract_ok": public_ok,
            "audit_complete": False,
            "audit_requires_repair": False,
            "repeated_failure": repeated,
        }

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
        navigator_path = _state.get("navigator_report_path")
        if navigator_path:
            builder_context = (
                "# Public task and result protocol\n\n"
                + self.task_context
                + "\n\n# Navigator handoff\n\n"
                + require_handoff(self.workdir / navigator_path, "navigator")
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
            trigger="navigator_handoff" if navigator_path else "initial_task",
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
        self._clear_public_artifacts()
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
        update = self._execution_update(
            _state, start=execution_start, ok=eval_run.ok, n_exec=1
        )
        update["execution_start"] = execution_start
        if self.policy.full_team:
            update.update(
                {
                    "reviewer_report_path": "review_report.md",
                    "audit_complete": True,
                    "audit_requires_repair": self._review(
                        0, execution_start, role_prefix="reviewer"
                    ),
                }
            )
        return update

    def _review(
        self,
        round_index: int,
        latest_execution_start: int,
        *,
        role_prefix: str,
    ) -> bool:
        diagnostics = self.contract_diagnostics(self.session)
        context_parts = ["# Public task and result protocol\n\n" + self.task_context]
        navigator_report = self.workdir / "navigator_report.md"
        if navigator_report.is_file():
            context_parts.append(
                "# Navigator handoff\n\n"
                + require_handoff(navigator_report, "navigator")
            )
        context_parts.extend(
            [
                "# Evaluation implementation\n\n"
                + clip_text(
                    (self.workdir / self.config.eval_script).read_text(
                        errors="replace"
                    ),
                    12000,
                ),
                "# Latest public execution log\n\n"
                + clip_text(public_log(self.session, latest_execution_start), 12000),
                "# Deterministic public-contract audit\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics),
            ]
        )
        review_context = "\n\n".join(context_parts)
        key = f"{role_prefix}_{round_index}"
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
        report = (self.workdir / "review_report.md").read_text(errors="replace")
        return "REVIEW_STATUS: REPAIR_REQUIRED" in report

    def _node_audit(self, state: _RunState) -> dict[str, Any]:
        start = state["latest_execution_start"]
        if start is None:
            raise RuntimeError("audit requires a prior execution")
        requires_repair = self._review(
            state["round"], start, role_prefix="auditor"
        )
        return {
            "reviewer_report_path": "review_report.md",
            "audit_complete": True,
            "audit_requires_repair": requires_repair,
        }

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
        review_path = state.get("reviewer_report_path")
        if review_path:
            label = "Auditor" if self.policy.adaptive else "Independent reviewer"
            parts.append(
                f"# {label} audit\n\n"
                + require_handoff(self.workdir / review_path, label.lower())
            )
        navigator_path = state.get("navigator_report_path")
        if navigator_path:
            parts.extend(
                [
                    "# Navigator handoff\n\n"
                    + require_handoff(self.workdir / navigator_path, "navigator"),
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
        self._clear_public_artifacts()
        start = len(self.session.transcript)
        self.eval_executions_observed += 1
        stepped_run = config.execute_eval(self.session)
        self.roles[key]["errors"] = 0 if stepped_run.ok else 1
        self.roles[key]["command_indexes"] = [start + 1, len(self.session.transcript)]
        log = public_log(self.session, execution_start)
        self.session.write_file("reproducer_public_log.txt", log)

        reviewer_requires_repair = False
        if self.policy.full_team:
            reviewer_requires_repair = self._review(
                round_index, start, role_prefix="reviewer"
            )

        update = self._execution_update(
            state,
            start=start,
            ok=stepped_run.ok,
            n_exec=state["n_exec"] + 1,
        )
        update.update(
            {
                "round": round_index,
                "reviewer_report_path": (
                    "review_report.md" if self.policy.full_team else None
                ),
                "audit_complete": self.policy.full_team,
                "audit_requires_repair": reviewer_requires_repair,
            }
        )
        return update

    def _needs_adaptive_audit(self, state: _RunState) -> bool:
        if not self.policy.adaptive or state["audit_complete"]:
            return False
        if state["public_contract_ok"]:
            return bool(
                self.route_decision and self.route_decision.require_semantic_audit
            )
        return state["repeated_failure"] or state["failure_kind"] in {
            "semantic_mismatch",
            "unknown_failure",
        }

    def _decide(self, state: _RunState) -> str:
        """Stop, audit, or repair from public evidence and the shared budget."""
        if not self.policy.allow_repair:
            return "end"
        if self._needs_adaptive_audit(state):
            return "audit"
        if state["public_contract_ok"] and not state["audit_requires_repair"]:
            return "end"
        if state["round"] >= MAX_REPAIR_ROUNDS:
            return "end"
        return "repair"

    def _route_entry(self, _state: _RunState) -> str:
        if self.route_decision and self.route_decision.use_navigator:
            return "navigate"
        return "reproduce"

    # --- driver -----------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(_RunState)
        graph.add_node("reproduce", self._node_reproduce)
        graph.add_node("execute", self._node_execute)
        graph.add_node("repair", self._node_repair)
        graph.add_node("audit", self._node_audit)

        if self.policy.full_team:
            graph.add_node("navigate", self._node_navigate)
            graph.add_node("critique", self._node_critique)
            graph.set_entry_point("navigate")
            graph.add_edge("navigate", "reproduce")
            graph.add_edge("reproduce", "critique")
            graph.add_edge("critique", "execute")
        elif self.policy.adaptive:
            graph.add_node("route", lambda _state: {})
            graph.add_node("navigate", self._node_navigate)
            graph.set_entry_point("route")
            graph.add_conditional_edges(
                "route",
                self._route_entry,
                {"navigate": "navigate", "reproduce": "reproduce"},
            )
            graph.add_edge("navigate", "reproduce")
            graph.add_edge("reproduce", "execute")
        else:
            graph.set_entry_point("reproduce")
            graph.add_edge("reproduce", "execute")

        # Repair performs its execution internally, then returns to this routing
        # decision. Adaptive may insert an audit before generation or repair.
        destinations = {"audit": "audit", "repair": "repair", "end": END}
        graph.add_conditional_edges("execute", self._decide, destinations)
        graph.add_conditional_edges("audit", self._decide, destinations)
        graph.add_conditional_edges("repair", self._decide, destinations)
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
    elif policy.adaptive:
        required = []
        if "navigator" in pipe.roles:
            required.append("navigator_report.md")
        if any(name.startswith("auditor_") for name in pipe.roles):
            required.append("review_report.md")
        handoff_requirement = all((pipe.workdir / name).exists() for name in required)
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
        routing=pipe.route_decision.as_dict() if pipe.route_decision else None,
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
