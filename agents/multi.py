"""Multi-agent reproduction (the supporting component) + two ablations.

Lead decomposes a multi-result task into N sub-tasks; each Reproducer handles one
in an ISOLATED context (its own session, its own message history, **its own LLM
client** so token/cost accounting is per-agent and thread-safe); a Verifier
checks each deterministically.

Two honest questions the design poses:
  1. Does isolation help vs one agent doing all N in a single shared context?
     (measured: success, per-agent context size, steps, cost)
  2. Since the sub-tasks are independent, does running them CONCURRENTLY cut
     wall-clock — or does single-box resource contention (MPS/RAM) eat the win?
     (measured: parallel vs serial wall-clock; reported as-is)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from agent.llm import LLM, ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run

LLMFactory = Callable[[], LLM]


@dataclass
class SubResult:
    model: str
    expected: float
    actual: float | None
    matched: bool
    steps: int
    errors: int
    cost_yuan: float = 0.0
    ctx_chars: int = 0


def _task(model: str, repo: str, dataset: str) -> str:
    # Mirror the proven single-oracle wording (run_repro.build_task) VERBATIM so
    # the isolated Reproducer inherits its reliability. Two lessons learned the
    # hard way: (1) pass the **full repo URL**, not the owner/repo slug — the slug
    # made the agent dither (endlessly explore the hub cache) instead of acting;
    # (2) NO `target` field here — each Reproducer is isolated and emits exactly
    # one result, so target-disambiguation (and the "multi-model" cue the model
    # then under-applies) is unnecessary. Target lives only in the single-context
    # baseline, which must label 3 results in one transcript.
    return (
        f"Reproduce the published top-1 accuracy (in percent) of the model "
        f"'{model}' from the torch.hub repository '{repo}' on the {dataset}. Use "
        f"the machine-readable metric id 'top1_accuracy' in REPRO_RESULT."
    )


def lead_decompose(manifest: dict) -> list[dict]:
    """Lead: split the multi-result task into per-target sub-tasks.

    Deterministic here (sub-targets are listed in the manifest) — the component
    being *measured* is the isolation/concurrency, not the decomposition, so we
    keep this honest and thin rather than dressing a trivial split up as an LLM
    call.
    """
    return list(manifest["subtargets"])


def run_multi(
    manifest: dict,
    session_root,
    repro_py,
    make_llm: LLMFactory = ChatLLM,
    *,
    parallel: bool = True,
    max_workers: int = 3,
) -> dict:
    """Run each sub-target in an isolated Reproducer. ``parallel`` runs them on a
    thread pool (LLM I/O + subprocess eval both release the GIL); serial is the
    baseline the concurrency ablation compares against."""
    repo = manifest["repo"]  # full URL — the slug makes the agent dither (see _task)
    dataset, tol = manifest["dataset"]["name"], float(manifest["tolerance"])
    num_examples = int(manifest["dataset"]["num_examples"])
    subs = lead_decompose(manifest)

    def reproduce_one(st: dict) -> SubResult:
        llm = make_llm()  # per-agent client → isolated, thread-safe usage
        session = Session(session_root / f"multi_{st['model']}", venv_python=repro_py, default_timeout=300)
        r = run_agent(_task(st["model"], repo, dataset), session, llm, max_steps=12)
        v = verify_run(
            session.transcript,
            session.workdir,
            expected=float(st["expected"]),
            tolerance=tol,
            metric=manifest["metric"],
            expected_num_examples=num_examples,
            target=None,  # isolated agent → exactly one result; no disambiguation needed
        )
        return SubResult(
            st["model"], st["expected"], v.actual, v.match, r.steps, r.errors,
            cost_yuan=r.usage.get("cost_yuan", 0.0), ctx_chars=r.peak_ctx_chars,
        )

    start = time.monotonic()
    if parallel:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(subs))) as pool:
            results = list(pool.map(reproduce_one, subs))
    else:
        results = [reproduce_one(st) for st in subs]
    wall_s = time.monotonic() - start

    return {
        "mode": "multi-parallel" if parallel else "multi-serial",
        "n_agents": len(subs), "results": results,
        "matched": sum(x.matched for x in results),
        # isolation property: the largest context any *single* agent reached.
        "max_ctx_chars": max((x.ctx_chars for x in results), default=0),
        "total_steps": sum(x.steps for x in results),
        "total_cost_yuan": round(sum(x.cost_yuan for x in results), 4),
        "wall_s": round(wall_s, 1),
    }


def run_single(manifest: dict, session_root, repro_py, make_llm: LLMFactory = ChatLLM) -> dict:
    """One agent does all N sub-targets in a single shared context."""
    repo = manifest["repo"]  # full URL (see _task)
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
    llm = make_llm()
    session = Session(session_root / "single", venv_python=repro_py, default_timeout=600)
    start = time.monotonic()
    r = run_agent(task, session, llm, max_steps=20)
    wall_s = time.monotonic() - start

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
            SubResult(st["model"], st["expected"], verdict.actual, verdict.match,
                      r.steps, r.errors, ctx_chars=r.peak_ctx_chars)
        )
    return {
        "mode": "single", "n_agents": 1, "results": results,
        "matched": sum(x.matched for x in results),
        "max_ctx_chars": r.peak_ctx_chars, "total_steps": r.steps,
        "total_cost_yuan": round(r.usage.get("cost_yuan", 0.0), 4),
        "wall_s": round(wall_s, 1),
    }
