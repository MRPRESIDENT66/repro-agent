"""Reliability harness — run a manifest's hint-light task N times.

    python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5             # tool calls, serial
    python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5 --no-fc     # text-protocol ablation
    python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5 --workers 3 # run the N trials concurrently

Reports staged pass rates + cost + wall-clock. Each trial is fully isolated (own
Session, own ChatLLM), so `--workers K` just runs them on a thread pool. Every
claimed match is gated by an anti-hardcode check: the eval script must actually
load the dataset and argmax over it.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from agent.llm import ChatLLM
from agent.loop import run_agent
from exec.session import Session
from verify.check import verify_run
from run_repro import build_task

ROOT = Path(__file__).resolve().parent
REPRO_PY = ROOT / ".venv-oracle" / "bin" / "python"


def _arg(argv: list[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default


def main() -> None:
    argv = sys.argv[1:]
    use_tools = "--no-fc" not in argv
    workers = int(_arg(argv, "--workers", "1"))
    # positional args = everything that isn't a flag or the value following --workers
    pos = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        if a.startswith("--"):
            continue
        pos.append(a)
    manifest = pos[0] if pos else "evals/benchmark/cifar10_resnet20.yaml"
    N = int(pos[1]) if len(pos) > 1 else 5

    m = yaml.safe_load((ROOT / manifest).read_text())
    task = build_task(m)
    expected, tol = float(m["target"]["expected"]), float(m["target"]["tolerance"])
    name = Path(manifest).stem

    def trial(i: int) -> dict:
        ws = ROOT / f"workspaces/rel_{name}_{i}"
        shutil.rmtree(ws, ignore_errors=True)
        session = Session(ws, venv_python=REPRO_PY, default_timeout=400)
        r = run_agent(task, session, ChatLLM(), max_steps=20, use_tools=use_tools)
        v = verify_run(
            session.transcript, session.workdir,
            expected=expected, tolerance=tol,
            metric=m["target"]["metric"],
            expected_num_examples=int(m["dataset"]["num_examples"]),
        )
        row = {"i": i, "matched": v.match, "metric": v.actual is not None,
               "steps": r.steps, "errors": r.errors, "fmt": r.format_errors,
               "actual": v.actual, "cost": r.usage.get("cost_yuan", 0.0)}
        print(f"run {i}: matched={v.match} actual={v.actual} steps={r.steps} "
              f"errors={r.errors} fmt_errors={r.format_errors} cost=¥{row['cost']}")
        return row

    start = time.monotonic()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = sorted(pool.map(trial, range(N)), key=lambda x: x["i"])
    else:
        rows = [trial(i) for i in range(N)]
    wall_s = time.monotonic() - start

    proto = "tool_calls" if use_tools else "text"
    print(f"\n===== {name}: reliability over {N} runs "
          f"(deepseek-chat, {proto}, workers={workers}) =====")
    for stage in ("metric", "matched"):
        c = sum(row[stage] for row in rows)
        print(f"  {stage:10} {c}/{N} = {c/N:.0%}")
    print(f"  avg steps={sum(r['steps'] for r in rows)/N:.1f}  "
          f"avg errors={sum(r['errors'] for r in rows)/N:.1f}  "
          f"avg fmt_errors={sum(r['fmt'] for r in rows)/N:.2f}  "
          f"avg cost=¥{sum(r['cost'] for r in rows)/N:.4f}  "
          f"wall={wall_s:.0f}s")


if __name__ == "__main__":
    main()
