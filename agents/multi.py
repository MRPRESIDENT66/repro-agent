"""Multi-agent reproduction (the supporting component) + the isolation ablation.

Lead decomposes a multi-result task into N sub-tasks; each Reproducer handles one
in an ISOLATED context (its own session + its own message history); a Verifier
checks each deterministically. The honest question the design poses: does the
isolation actually help, vs one agent doing all N in a single shared context?

Measured: success (matched sub-results), context size (the design's isolation
property: per-agent vs shared), and cost (LLM steps). Reported as-is — multi-agent
is expected to keep per-agent context smaller, NOT necessarily to win on success.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.llm import LLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run


@dataclass
class SubResult:
    model: str
    expected: float
    actual: float | None
    matched: bool
    steps: int
    errors: int


def _task(model: str, repo: str, dataset: str) -> str:
    return (
        f"Reproduce the published top-1 accuracy (in percent) of model '{model}' "
        f"from the torch.hub repository '{repo}' on the {dataset}. Use metric id "
        f"'top1_accuracy' and target '{model}' in REPRO_RESULT."
    )


def _ctx_size(transcript) -> tuple[int, int]:
    return len(transcript), sum(len(m["content"]) for m in transcript)


def lead_decompose(manifest: dict) -> list[dict]:
    """Lead: split the multi-result task into per-target sub-tasks.

    Deterministic here (sub-targets are listed in the manifest) — the component
    being *measured* is the isolation, not the decomposition, so we keep this
    honest and thin rather than dressing a trivial split up as an LLM call.
    """
    return list(manifest["subtargets"])


def run_multi(manifest: dict, session_root, repro_py, llm: LLM) -> dict:
    repo = manifest["repo"].rstrip("/").split("/")[-2] + "/" + manifest["repo"].rstrip("/").split("/")[-1]
    dataset, tol = manifest["dataset"]["name"], float(manifest["tolerance"])
    num_examples = int(manifest["dataset"]["num_examples"])
    subs = lead_decompose(manifest)

    results, max_msgs, max_chars, total_steps = [], 0, 0, 0
    for st in subs:
        session = Session(session_root / f"multi_{st['model']}", venv_python=repro_py, default_timeout=300)
        r = run_agent(_task(st["model"], repo, dataset), session, llm, max_steps=12)
        v = verify_run(
            session.transcript,
            session.workdir,
            expected=float(st["expected"]),
            tolerance=tol,
            metric=manifest["metric"],
            expected_num_examples=num_examples,
            target=st["model"],
        )
        results.append(SubResult(st["model"], st["expected"], v.actual, v.match, r.steps, r.errors))
        msgs, chars = _ctx_size(r.transcript)
        max_msgs, max_chars = max(max_msgs, msgs), max(max_chars, chars)
        total_steps += r.steps
    return {
        "mode": "multi", "n_agents": len(subs), "results": results,
        "matched": sum(x.matched for x in results), "max_ctx_msgs": max_msgs,
        "max_ctx_chars": max_chars, "total_steps": total_steps,
    }


def run_single(manifest: dict, session_root, repro_py, llm: LLM) -> dict:
    repo = manifest["repo"].rstrip("/").split("/")[-2] + "/" + manifest["repo"].rstrip("/").split("/")[-1]
    dataset, tol = manifest["dataset"]["name"], float(manifest["tolerance"])
    num_examples = int(manifest["dataset"]["num_examples"])
    subs = manifest["subtargets"]
    models = ", ".join(f"'{s['model']}'" for s in subs)
    task = (
        f"Reproduce the published top-1 accuracy (in percent) of ALL of these "
        f"models — {models} — from the torch.hub repository '{repo}' on the "
        f"{dataset}. Print one REPRO_RESULT per model using metric id "
        f"'top1_accuracy' and that model name as target."
    )
    session = Session(session_root / "single", venv_python=repro_py, default_timeout=600)
    r = run_agent(task, session, llm, max_steps=20)

    results = []
    for st in subs:
        verdict = verify_run(
            session.transcript,
            session.workdir,
            expected=float(st["expected"]),
            tolerance=tol,
            metric=manifest["metric"],
            expected_num_examples=num_examples,
            target=st["model"],
        )
        results.append(
            SubResult(st["model"], st["expected"], verdict.actual, verdict.match, r.steps, r.errors)
        )
    msgs, chars = _ctx_size(r.transcript)
    return {
        "mode": "single", "n_agents": 1, "results": results,
        "matched": sum(x.matched for x in results), "max_ctx_msgs": msgs,
        "max_ctx_chars": chars, "total_steps": r.steps,
    }
