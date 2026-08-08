"""Adaptive blind multi-agent reproduction pipeline.

Read this file top to bottom to see the whole system:

- ``provision_workspace``  — set up the blind sandbox + execution session.
- ``ReproductionPipeline`` — start short, then add investigation/review and
                             full collaboration only when evidence requires it.
- ``run_oracle``           — thin driver: run the pipeline, verify, emit.

The agent never sees the hidden target; an independent verifier recomputes the
metric from per-sample artifacts. Each role starts from a fresh LLM context.
"""

from __future__ import annotations

import json
import shutil
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
    RoleSynthesisError,
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
    collaboration_level: str
    execution_owner: str
    navigator_report_path: str | None
    reviewer_report_path: str | None
    execution_start: int | None
    latest_execution_start: int | None
    reviewed_execution_start: int | None
    n_exec: int
    public_contract_ok: bool | None
    review_requires_repair: bool


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

    Construction provisions the blind sandbox; :meth:`run` drives one adaptive
    graph. Simple tasks start with Reproducer alone. The first failure adds
    Navigator and Reviewer; a repeated failure or semantic-risk route escalates
    to the complete Navigator/Reproducer/Critic/Reviewer/Repair collaboration.
    The private verifier runs once after this workflow. LangGraph State holds
    routing metadata while ``workdir`` retains auditable artifacts.
    """

    def __init__(self, config: OracleConfig) -> None:
        self.config = config
        self.prompts = GENERIC_PROMPTS
        self.task_context = generic_task_context(config)
        self.code_validator = make_generic_code_validator(config)
        self.contract_diagnostics = make_generic_contract_diagnostics(config)
        self.synthesis_instruction = (
            f"Return only the complete executable source code for {config.eval_script}. "
            "The program must produce the public result artifact when executed. "
            "Do not return the contents of predictions or result files. Your first "
            "non-whitespace token must be Python source or a ```python fence; do not "
            "describe another search, plan, or future action."
        )

        self.workdir = config.workdir
        self.artifact_dir = config.artifact_dir

        self.session = provision_workspace(config, self.artifact_dir)
        self.role_deps = RoleDeps(
            llm_factory=ChatLLM,
            search_fn=search_repo,
            snippet_fn=relevant_snippet,
        )

        self.roles: dict[str, dict] = {}
        self.rag: dict[str, dict] = {}
        self.route_decision: RouteDecision | None = None
        self.collaboration_level = "short"
        self.workflow_error: str | None = None
        self.workflow_warnings: list[str] = []
        self.failure_classes: list[dict[str, str | None]] = []
        # Kept outside LangGraph State so an exception after execution (for
        # example, Reviewer synthesis failure) cannot erase the observed count.
        self.eval_executions_observed = 0
        self.run_state: _RunState = self._initial_state()

    def _initial_state(self) -> _RunState:
        return {
            "round": 0,
            "collaboration_level": "short",
            "execution_owner": "reproducer",
            "navigator_report_path": None,
            "reviewer_report_path": None,
            "execution_start": None,
            "latest_execution_start": None,
            "reviewed_execution_start": None,
            "n_exec": 0,
            "public_contract_ok": None,
            "review_requires_repair": False,
        }

    # --- shared helpers ---------------------------------------------------

    def passed(self) -> bool:
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

    def _run_evaluation(self, state: _RunState, *, owner: str) -> dict[str, Any]:
        """Execute the current script and return the state derived from that run."""
        self._sync_eval_file()
        self._clear_public_artifacts()
        start = len(self.session.transcript)
        self.eval_executions_observed += 1
        result = self.config.execute_eval(self.session)

        self.roles[owner]["errors"] = 0 if result.ok else 1
        self.roles[owner]["command_indexes"] = [
            start + 1,
            len(self.session.transcript),
        ]

        history_start = state["execution_start"]
        if history_start is None:
            history_start = start
        self.session.write_file(
            "reproducer_public_log.txt",
            public_log(self.session, history_start),
        )

        update = self._execution_update(
            start=start,
            n_exec=state["n_exec"] + 1,
        )
        if state["execution_start"] is None:
            update["execution_start"] = start
        return update

    def _execution_update(self, *, start: int, n_exec: int) -> dict[str, Any]:
        return {
            "latest_execution_start": start,
            "n_exec": n_exec,
            "public_contract_ok": self.passed(),
            "reviewed_execution_start": None,
            "review_requires_repair": False,
        }

    # --- stages -----------------------------------------------------------

    def _routing_context(self) -> str:
        if self.route_decision is None:
            return ""
        return "\n\n" + self.route_decision.downstream_context()

    def _node_route(self, _state: _RunState) -> dict[str, Any]:
        self.route_decision, self.roles["router"] = route_task(
            self.config,
            self.workdir,
            self.artifact_dir,
            llm_factory=self.role_deps.llm_factory,
        )
        if self.route_decision.require_semantic_review:
            level = "full"
        elif self.route_decision.use_navigator:
            level = "assisted"
        else:
            level = "short"
        self.collaboration_level = level
        return {"collaboration_level": level}

    def _node_navigate(self, _state: _RunState) -> dict[str, Any]:
        context = self.task_context + self._routing_context()
        latest_start = _state["latest_execution_start"]
        if latest_start is not None:
            diagnostics = self.contract_diagnostics(self.session)
            context += (
                "\n\n# Failed implementation\n\n"
                + clip_text(
                    (self.workdir / self.config.eval_script).read_text(
                        errors="replace"
                    ),
                    10000,
                )
                + "\n\n# First execution failure\n\n"
                + clip_text(public_log(self.session, latest_start), 10000)
                + "\n\n# Public diagnostics\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics)
            )
        self.roles["navigator"], self.rag["navigator"] = run_rag_role(
            name="navigator",
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            instruction=self.prompts.navigator,
            context=context,
            output_path=self.workdir / "navigator_report.md",
            submit_name="submit_handoff",
            submit_description="Submit the source-grounded Navigator handoff.",
            validator=validate_report,
            trigger=(
                "first_execution_failure"
                if _state["latest_execution_start"] is not None
                else "initial_task"
            ),
            max_steps=7,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )
        return {"navigator_report_path": "navigator_report.md"}

    def _node_reproduce(self, _state: _RunState) -> dict[str, Any]:
        navigator_path = _state.get("navigator_report_path")
        public_context = self.task_context + self._routing_context()
        if navigator_path:
            builder_context = (
                "# Public task and result protocol\n\n"
                + public_context
                + "\n\n# Navigator handoff\n\n"
                + require_handoff(self.workdir / navigator_path, "navigator")
            )
        else:
            builder_context = public_context

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
        return {"execution_owner": "reproducer"}

    def _node_critique(self, state: _RunState) -> dict[str, Any]:
        """Run the full-path preflight Critic over the current program."""
        context_parts = [
            "# Public task and result protocol\n\n"
            + self.task_context
            + self._routing_context(),
            "# Current evaluation script\n\n"
            + (self.workdir / self.config.eval_script).read_text(errors="replace"),
        ]
        navigator_path = state.get("navigator_report_path")
        if navigator_path:
            context_parts.append(
                "# Navigator handoff\n\n"
                + require_handoff(self.workdir / navigator_path, "navigator")
            )
        reviewer_path = state.get("reviewer_report_path")
        if reviewer_path:
            context_parts.append(
                "# Latest Reviewer diagnosis\n\n"
                + require_handoff(self.workdir / reviewer_path, "reviewer")
            )

        key = "critic" if state["n_exec"] == 0 else f"critic_{state['n_exec']}"
        self.roles[key], self.rag[key] = run_rag_role(
            name=key,
            workdir=self.workdir,
            artifact_dir=self.artifact_dir,
            session=self.session,
            instruction=self.prompts.critic,
            context="\n\n".join(context_parts),
            output_path=self.workdir / self.config.eval_script,
            submit_name="submit_code",
            submit_description=f"Submit the complete reviewed {self.config.eval_script}.",
            validator=self.code_validator,
            trigger=(
                "semantic_risk_preflight"
                if state["n_exec"] == 0
                else "repeated_failure_escalation"
            ),
            max_steps=7,
            synthesis_instruction=self.synthesis_instruction,
            synthesis_attempts=5,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            deps=self.role_deps,
        )
        return {"execution_owner": key}

    def _node_execute(self, _state: _RunState) -> dict[str, Any]:
        return self._run_evaluation(_state, owner=_state["execution_owner"])

    def _node_review(self, state: _RunState) -> dict[str, Any]:
        latest_execution_start = state["latest_execution_start"]
        if latest_execution_start is None:
            raise RuntimeError("review requires a prior execution")

        round_index = state["n_exec"] - 1
        diagnostics = self.contract_diagnostics(self.session)
        context_parts = [
            "# Public task and result protocol\n\n"
            + self.task_context
            + self._routing_context()
        ]
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
                "# Deterministic public-contract check\n\n"
                + "\n".join(f"- {issue}" for issue in diagnostics),
            ]
        )
        review_context = "\n\n".join(context_parts)
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
            submit_description="Submit the source-grounded execution review.",
            validator=validate_review,
            trigger="execution_result" if round_index == 0 else "repair_execution_result",
            max_steps=6,
            max_queries=2,
            search_extra_exclude=self.config.search_extra_exclude,
            allow_runtime_probe=True,
            synthesis_instruction=(
                "Return only the complete Markdown review report. Include the four "
                "required source-evidence lines with backticked repository paths, "
                "then end with exactly REVIEW_STATUS: PASS or REVIEW_STATUS: "
                "REPAIR_REQUIRED. Do not return Python source code."
            ),
            deps=self.role_deps,
        )
        report = (self.workdir / "review_report.md").read_text(errors="replace")
        requires_repair = "REVIEW_STATUS: REPAIR_REQUIRED" in report
        level = (
            "assisted"
            if state["collaboration_level"] == "short"
            else state["collaboration_level"]
        )
        self.collaboration_level = level
        return {
            "collaboration_level": level,
            "reviewer_report_path": "review_report.md",
            "reviewed_execution_start": latest_execution_start,
            "review_requires_repair": requires_repair,
        }

    def _node_escalate(self, _state: _RunState) -> dict[str, Any]:
        """Promote a repeatedly failing attempt to the complete collaboration path."""
        self.collaboration_level = "full"
        return {"collaboration_level": "full"}

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
            "# Public task and result protocol\n\n"
            + self.task_context
            + self._routing_context(),
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
        reviewer_path = state.get("reviewer_report_path")
        if reviewer_path:
            parts.append(
                "# Reviewer diagnosis\n\n"
                + require_handoff(self.workdir / reviewer_path, "reviewer")
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
        parts.append("# Deterministic public-contract check\n\n" + diagnostic_text)
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
        try:
            self.roles[key], self.rag[key] = run_rag_role(
                name=key,
                workdir=self.workdir,
                artifact_dir=self.artifact_dir,
                session=self.session,
                instruction=self.prompts.repair.replace(
                    "{round_index}", str(round_index)
                ),
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
                    "submit_patch",
                    "Patch the current eval script with exact old/new replacements.",
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
        except RoleSynthesisError as exc:
            self.roles[key], self.rag[key] = exc.role, exc.rag
            if not state["public_contract_ok"]:
                raise
            self.workflow_warnings.append(
                f"{key} synthesis failed after a public-contract-valid artifact; "
                "kept the last measured artifact for private verification"
            )
            return {
                "round": round_index,
                "review_requires_repair": False,
            }

        update = self._run_evaluation(state, owner=key)
        update.update(
            {
                "round": round_index,
                "execution_owner": key,
                "reviewer_report_path": None,
                "reviewed_execution_start": None,
                "review_requires_repair": False,
            }
        )
        return update

    def _current_execution_reviewed(self, state: _RunState) -> bool:
        return (
            state["latest_execution_start"] is not None
            and state["reviewed_execution_start"] == state["latest_execution_start"]
        )

    def _decide(self, state: _RunState) -> str:
        """Select the next adaptive stage from evidence and collaboration level."""
        full = state["collaboration_level"] == "full"
        reviewed = self._current_execution_reviewed(state)

        if state["public_contract_ok"]:
            if full and not reviewed:
                return "review"
            if not state["review_requires_repair"]:
                return "end"

        if state["n_exec"] >= MAX_REPAIR_ROUNDS + 1:
            return "end"

        if not state["public_contract_ok"]:
            if state["navigator_report_path"] is None:
                return "navigate"
            if not reviewed:
                return "review"
            if state["n_exec"] >= 2 and not full:
                return "escalate"

        return "repair"

    def _route_entry(self, _state: _RunState) -> str:
        if _state["collaboration_level"] == "full" or (
            self.route_decision and self.route_decision.use_navigator
        ):
            return "navigate"
        return "reproduce"

    def _after_navigate(self, state: _RunState) -> str:
        return "review" if state["latest_execution_start"] is not None else "reproduce"

    def _after_reproduce(self, state: _RunState) -> str:
        return "critique" if state["collaboration_level"] == "full" else "execute"

    # --- driver -----------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(_RunState)
        graph.add_node("route", self._node_route)
        graph.add_node("navigate", self._node_navigate)
        graph.add_node("reproduce", self._node_reproduce)
        graph.add_node("critique", self._node_critique)
        graph.add_node("execute", self._node_execute)
        graph.add_node("repair", self._node_repair)
        graph.add_node("review", self._node_review)
        graph.add_node("escalate", self._node_escalate)

        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            self._route_entry,
            {"navigate": "navigate", "reproduce": "reproduce"},
        )
        graph.add_conditional_edges(
            "navigate",
            self._after_navigate,
            {"review": "review", "reproduce": "reproduce"},
        )
        graph.add_conditional_edges(
            "reproduce",
            self._after_reproduce,
            {"critique": "critique", "execute": "execute"},
        )
        graph.add_edge("critique", "execute")
        graph.add_edge("escalate", "critique")

        # Repair performs its execution internally, then returns to this routing
        # decision. Failures progressively add investigation, review, and full
        # preflight collaboration while sharing one execution budget.
        destinations = {
            "navigate": "navigate",
            "review": "review",
            "escalate": "escalate",
            "repair": "repair",
            "end": END,
        }
        graph.add_conditional_edges("execute", self._decide, destinations)
        graph.add_conditional_edges("review", self._decide, destinations)
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


def run_oracle(config: OracleConfig) -> None:
    pipe = ReproductionPipeline(config).run()
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
    required = []
    if "navigator" in pipe.roles:
        required.append("navigator_report.md")
    if any(name.startswith("reviewer_") for name in pipe.roles):
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
        pipeline="adaptive",
        n_exec=pipe.eval_executions_observed,
        roles=pipe.roles,
        rag=pipe.rag,
        workflow_error=pipe.workflow_error,
        rag_requirement=rag_requirement,
        handoff_requirement=handoff_requirement,
        collaboration_pass=collaboration_pass,
        public_evidence_found=pipe.passed(),
        public_contract_diagnostics=pipe.contract_diagnostics(session),
        verdict=verdict,
        total_commands=len(session.transcript),
        probe_transcript=probe_transcript,
        failure_classes=pipe.failure_classes,
        workflow_warnings=pipe.workflow_warnings,
        routing=pipe.route_decision.as_dict() if pipe.route_decision else None,
        collaboration_level=pipe.collaboration_level,
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
