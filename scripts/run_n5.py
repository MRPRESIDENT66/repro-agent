#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "n5"
LOCK_PATH = LOG_DIR / "run.lock"


@dataclass(frozen=True)
class RunSpec:
    group: str
    task: str
    script: str
    attempt_env: str
    attempt: str
    pipeline: str

    @property
    def log_name(self) -> str:
        return f"{self.group}__{self.task}__{self.pipeline}__{self.attempt}.log"


E1_TASKS = [
    ("distilbert", "run_distilbert_multi_rag.py", "DISTILBERT_ATTEMPT"),
    ("detectors_rn18", "run_detectors_resnet18_cifar100_multi_rag.py", "DETECTORS_ATTEMPT"),
    ("detectors_vgg16", "run_detectors_vgg16_cifar10_multi_rag.py", "DETECTORS_ATTEMPT"),
    ("mmpretrain", "run_mmpretrain_multi_rag.py", "MMPRETRAIN_ATTEMPT"),
    ("openood", "run_openood_multi_rag.py", "OPENOOD_MULTI_RAG_ATTEMPT"),
    ("robustbench", "run_robustbench_multi_rag.py", "ROBUSTBENCH_ATTEMPT"),
]
E2_TASKS = [
    ("distilbert", "run_distilbert_multi_rag.py", "DISTILBERT_ATTEMPT"),
    ("detectors_rn18", "run_detectors_resnet18_cifar100_multi_rag.py", "DETECTORS_ATTEMPT"),
]
PIPELINES = ["solo", "solo-repair", "full"]


def build_specs(seeds: int, include_robustbench: bool) -> list[RunSpec]:
    specs: list[RunSpec] = []
    e1_tasks = E1_TASKS if include_robustbench else [t for t in E1_TASKS if t[0] != "robustbench"]
    for seed in range(1, seeds + 1):
        for task, script, env_name in e1_tasks:
            specs.append(
                RunSpec(
                    group="e1_n5",
                    task=task,
                    script=script,
                    attempt_env=env_name,
                    attempt=f"e1_n5_s{seed}",
                    pipeline="full",
                )
            )
        for task, script, env_name in E2_TASKS:
            for pipeline in PIPELINES:
                specs.append(
                    RunSpec(
                        group="e2_n5",
                        task=task,
                        script=script,
                        attempt_env=env_name,
                        attempt=f"e2_n5_s{seed}_{pipeline}",
                        pipeline=pipeline,
                    )
                )
    return specs


def result_path(spec: RunSpec) -> Path:
    suffix = "" if spec.pipeline == "full" else f"__{spec.pipeline}"
    if spec.task == "distilbert":
        return ROOT / "evals" / "runs" / f"distilbert_sst2_multi_rag_{spec.attempt}{suffix}" / "result.json"
    if spec.task == "detectors_rn18":
        return ROOT / "evals" / "runs" / f"detectors_resnet18_cifar100_multi_rag_{spec.attempt}{suffix}" / "result.json"
    if spec.task == "detectors_vgg16":
        return ROOT / "evals" / "runs" / f"detectors_vgg16_cifar10_multi_rag_{spec.attempt}{suffix}" / "result.json"
    if spec.task == "mmpretrain":
        return ROOT / "evals" / "runs" / f"mmpretrain_resnet18_multi_rag_{spec.attempt}{suffix}" / "result.json"
    if spec.task == "openood":
        return ROOT / "evals" / "runs" / f"openood_ebo_multi_rag_{spec.attempt}{suffix}" / "result.json"
    if spec.task == "robustbench":
        return ROOT / "evals" / "runs" / f"robustbench_carmon_{spec.attempt}{suffix}" / "result.json"
    raise ValueError(spec.task)


def run_one(spec: RunSpec, force: bool) -> int:
    result = result_path(spec)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / spec.log_name
    if result.exists() and not force:
        print(f"[skip] {spec.group} {spec.task} {spec.pipeline} {spec.attempt} -> {result.relative_to(ROOT)}", flush=True)
        return 0

    env = os.environ.copy()
    env["PIPELINE"] = spec.pipeline
    env[spec.attempt_env] = spec.attempt
    cmd = [sys.executable, spec.script]
    print(f"[run] {spec.group} {spec.task} {spec.pipeline} {spec.attempt}", flush=True)
    print(f"      {' '.join(cmd)} > {log_path.relative_to(ROOT)}", flush=True)
    start = time.time()
    with log_path.open("w") as log:
        log.write(f"# {' '.join(cmd)}\n")
        log.write(f"# PIPELINE={spec.pipeline} {spec.attempt_env}={spec.attempt}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    print(f"[done] exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path.relative_to(ROOT)}", flush=True)
    return proc.returncode


@contextmanager
def run_lock():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(
            f"another N=5 driver appears to be running; remove stale lock only if safe: {LOCK_PATH}"
        ) from exc
    with os.fdopen(fd, "w") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.write(f"started={time.time()}\n")
    try:
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1+E2 N=5 experiments.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="rerun even if result.json exists")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--include-robustbench", action="store_true", default=True)
    parser.add_argument("--no-robustbench", dest="include_robustbench", action="store_false")
    args = parser.parse_args()

    with run_lock():
        specs = build_specs(args.seeds, args.include_robustbench)
        print(f"total specs: {len(specs)}", flush=True)
        failures = 0
        for index, spec in enumerate(specs, start=1):
            print(f"\n[{index}/{len(specs)}]", flush=True)
            code = run_one(spec, args.force)
            if code != 0:
                failures += 1
                if args.stop_on_fail:
                    break
        print(f"\nfinished specs={len(specs)} process_failures={failures}", flush=True)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
