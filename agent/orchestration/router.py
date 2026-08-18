"""One-call LLM risk planning with deterministic routing safeguards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from agent.runtime.llm import ChatLLM
from agent.types import OracleConfig


SUBMIT_ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_route",
        "description": "Submit the task route and concrete downstream risk checklist.",
        "parameters": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["short", "assisted", "full"],
                    "description": (
                        "short = Reproducer only; assisted = Navigator then "
                        "Reproducer; full = Navigator, Reproducer, Critic, and Reviewer."
                    ),
                },
                "reasons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
                "review_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
            },
            "required": [
                "route",
                "reasons",
                "risk_flags",
                "review_requirements",
            ],
        },
    },
}

ROUTER_PROMPT = """You are the one-step risk Router for a blind ML reproduction task.
You cannot inspect the repository and must not solve or code the task. Call
submit_route exactly once. Select exactly one initial route: `short` for a direct
Reproducer path, `assisted` when source navigation is needed first, or `full` when
source navigation plus pre-execution Critic and post-execution Reviewer checks are
needed. Then produce only the 3-4 highest-priority risks and checks that downstream
agents can verify from repository evidence.

Flag risks such as model/data identity, preprocessing, score polarity, label
mapping, grouped aggregation, metric units, checkpoint coverage, and optional
dependency imports. Review requirements must be concrete falsifiable checks, not
generic advice. For OOD/AUROC tasks, explicitly distinguish repository confidence
direction from the public output convention and require proof of which group
should receive larger submitted scores. Do not prescribe raw versus negated
scores or assert a sign before observing repository and public-artifact evidence.

Use the short path for ordinary single-model classification when the public task
already provides a model card, cached model/data, exact split, output schema, and
standard accuracy. Zero Python files is expected for such a prepared task and is
not a reason to add Navigator or semantic review. Reserve semantic review for
meaningful ambiguity such as score direction, grouped aggregation, adversarial
attack configuration, similarity metrics, or conflicting repository evidence."""


@dataclass(frozen=True)
class RouteDecision:
    route: str
    repository_python_files: int
    reasons: tuple[str, ...]
    risk_flags: tuple[str, ...]
    review_requirements: tuple[str, ...]
    llm_route_valid: bool

    @property
    def use_navigator(self) -> bool:
        """Internal graph compatibility for routes that begin with research."""
        return self.route in {"assisted", "full"}

    @property
    def require_semantic_review(self) -> bool:
        """Internal graph compatibility for the complete collaboration route."""
        return self.route == "full"

    def as_dict(self) -> dict:
        return asdict(self)

    def downstream_context(self) -> str:
        risks = "\n".join(f"- {item}" for item in self.risk_flags) or "- none"
        checks = (
            "\n".join(f"- {item}" for item in self.review_requirements)
            or "- verify the public task and output contract"
        )
        return (
            "# Router risk plan\n\n"
            f"Risk flags:\n{risks}\n\n"
            f"Mandatory review requirements:\n{checks}\n\n"
            "Treat each requirement as unresolved until repository evidence proves it."
        )


_SEMANTIC_RISK_TERMS = (
    "auroc",
    "out-of-distribution",
    "ood",
    "energy score",
    "score direction",
    "aggregation",
    "adversarial",
    "autoattack",
    "robust accuracy",
    "cosine similarity",
    "spearman",
    "multiple checkpoints",
)


def _strings(value: object, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("route list fields must be arrays")
    items = tuple(
        _clip_item(str(item).strip()) for item in value if str(item).strip()
    )
    return items[:limit]


def _clip_item(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _remove_prescriptive_polarity_checks(
    requirements: tuple[str, ...], *, score_direction_risk: bool
) -> tuple[str, ...]:
    """Keep Router checks falsifiable instead of letting it pre-solve score sign."""
    if not score_direction_risk:
        return requirements
    prescriptive = (
        "polarity",
        "direction",
        "higher",
        "lower",
        "sign convention",
        "confidence",
        "negat",
        "raw energy",
        "raw score",
        "without inversion",
        "no inversion",
        "do not invert",
    )
    return tuple(
        requirement
        for requirement in requirements
        if not any(term in requirement.lower() for term in prescriptive)
    )


def _rule_plan(config: OracleConfig, workdir: Path) -> dict:
    public_text = f"{config.name}\n{config.task}\n{config.public_result_protocol}".lower()
    matched = [term for term in _SEMANTIC_RISK_TERMS if term in public_text]
    python_files = sum(1 for _ in workdir.rglob("*.py"))
    prepared_classification = (
        "model card" in public_text
        and python_files < 50
        and not matched
        and any(term in public_text for term in ("accuracy", "top-1"))
    )
    risks: list[str] = []
    requirements: list[str] = []

    if python_files >= 700:
        risks.append("large_repository")
        requirements.append("Trace the canonical evaluation entry and its transitive defaults.")
    if any(term in matched for term in ("auroc", "ood", "out-of-distribution", "energy score")):
        risks.extend(("score_direction", "grouped_aggregation"))
        requirements.extend(
            (
                "Prove whether ID or OOD samples must receive larger submitted scores; "
                "do not infer output polarity from a variable named confidence or energy.",
                "Verify the order of per-dataset and per-checkpoint AUROC aggregation.",
                "Trace the canonical evaluation preprocessor and verify its exact "
                "ordered operations, sizes, interpolation, and normalization constants.",
            )
        )
    if any(term in matched for term in ("adversarial", "autoattack", "robust accuracy")):
        risks.append("attack_semantics")
        requirements.append("Verify attack set, epsilon, restarts, and prediction ordering.")
    if any(term in matched for term in ("cosine similarity", "spearman")):
        risks.append("similarity_metric_semantics")
        requirements.append("Verify similarity pairing, normalization, ordering, and metric units.")

    return {
        "python_files": python_files,
        "matched": matched,
        "risks": tuple(dict.fromkeys(risks)),
        "requirements": tuple(dict.fromkeys(requirements)),
        "force_navigator": python_files >= 700 or bool(matched),
        "force_review": python_files >= 700 or bool(matched),
        "force_short": prepared_classification,
    }


def _write_router_artifacts(
    workdir: Path,
    artifact_dir: Path,
    transcript: list[dict],
    decision: RouteDecision,
) -> None:
    transcript_text = "".join(json.dumps(message) + "\n" for message in transcript)
    plan_text = json.dumps(decision.as_dict(), indent=2) + "\n"
    for directory in (workdir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "router_transcript.jsonl").write_text(transcript_text)
        (directory / "route_plan.json").write_text(plan_text)


def route_task(
    config: OracleConfig,
    workdir: Path,
    artifact_dir: Path,
    *,
    llm_factory: Callable[[], ChatLLM] = ChatLLM,
) -> tuple[RouteDecision, dict]:
    """Call an LLM once, then merge its plan with deterministic safeguards."""
    rules = _rule_plan(config, workdir)
    user_context = (
        f"# Public task\n{config.task}\n\n"
        f"# Public output contract\n{config.public_result_protocol}\n\n"
        f"# Cheap inventory\nPython files in workspace: {rules['python_files']}"
    )
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": user_context},
    ]
    llm = llm_factory()
    valid = False
    arguments: dict = {}
    error: str | None = None
    prompt_tokens = 0

    try:
        reply = llm.chat(messages, tools=[SUBMIT_ROUTE_TOOL])
        prompt_tokens = reply.prompt_tokens
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in reply.tool_calls
                ],
            }
        )
        if len(reply.tool_calls) != 1 or reply.tool_calls[0].name != "submit_route":
            raise ValueError("Router must call submit_route exactly once")
        arguments = reply.tool_calls[0].arguments
        llm_route = arguments.get("route")
        if llm_route not in {"short", "assisted", "full"}:
            raise ValueError("route must be short, assisted, or full")
        llm_reasons = _strings(arguments.get("reasons"), 3)
        llm_risks = _strings(arguments.get("risk_flags"), 4)
        llm_requirements = _strings(arguments.get("review_requirements"), 4)
        valid = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        llm_route = "short"
        llm_reasons = ()
        llm_risks = ()
        llm_requirements = ()

    reasons = list(llm_reasons)
    if rules["python_files"] >= 700:
        reasons.append(f"rule override: large repository ({rules['python_files']} Python files)")
    if rules["matched"]:
        reasons.append("rule override: semantic risk " + ", ".join(rules["matched"][:4]))
    if rules["force_short"]:
        reasons.append("rule override: prepared single-model classification task")
    if not reasons:
        reasons.append("rule fallback: small, explicit public task")

    llm_requirements = _remove_prescriptive_polarity_checks(
        llm_requirements,
        score_direction_risk="score_direction" in rules["risks"],
    )
    if rules["force_short"]:
        route = "short"
    elif rules["force_review"]:
        route = "full"
    elif rules["force_navigator"] and llm_route == "short":
        route = "assisted"
    else:
        route = llm_route

    decision = RouteDecision(
        route=route,
        repository_python_files=rules["python_files"],
        reasons=tuple(dict.fromkeys(reasons)),
        risk_flags=tuple(dict.fromkeys(rules["risks"] + llm_risks))[:6],
        review_requirements=tuple(
            dict.fromkeys(rules["requirements"] + llm_requirements)
        )[:5],
        llm_route_valid=valid,
    )
    if error:
        messages.append({"role": "user", "content": "Runtime rule fallback: " + error})
    else:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": reply.tool_calls[0].id,
                "content": "accepted route plan",
            }
        )
    _write_router_artifacts(workdir, artifact_dir, messages, decision)

    usage = llm.usage.as_dict()
    role = {
        "steps": 1,
        "errors": 0 if valid else 1,
        "format_errors": 0 if valid else 1,
        "artifact_submitted": valid,
        "usage": usage,
        "peak_ctx_tokens": prompt_tokens,
        "tool_counts": {"submit_route": 1} if valid else {},
        "command_indexes": [],
        "submission_trace": "route_plan.json",
        "runtime_probes": 0,
        "runtime_probe_recommended": False,
        "runtime_probe_hint": None,
        "probe_trace": None,
        "fallback_used": not valid,
    }
    return decision, role
