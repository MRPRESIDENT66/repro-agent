#!/usr/bin/env python3
"""Run the frozen-agent Sentence-Transformers held-out task five times."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / "logs" / "holdout_n5"
LOCK_PATH = LOG_DIR / "run.lock"
FROZEN_AGENT_COMMIT = "8fa152e"
SOURCE_COMMIT = "b2a9529cf6312d2b2a8ffa2b64d82fabc1571bd8"
DATASET_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

ADAPTER_FILES = (
    "evals/oracles/sentence_transformers_stsb.py",
    "evals/oracles/gold/stsb_test_scores.json",
    "run_sentence_transformers_stsb.py",
    "scripts/prepare_sentence_transformers_stsb.py",
    "scripts/run_holdout_n5.py",
    "tests/test_sentence_transformers_stsb.py",
)
FROZEN_AGENT_PATHS = ("agent", "retrieval", "exec", "verify")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frozen_agent() -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{FROZEN_AGENT_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", FROZEN_AGENT_COMMIT, "--", *FROZEN_AGENT_PATHS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if changed:
        raise SystemExit(
            "frozen agent/runtime paths changed after the holdout freeze:\n" + changed
        )
    source_head = subprocess.run(
        ["git", "-C", str(ROOT / "repos" / "sentence-transformers"), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if source_head != SOURCE_COMMIT:
        raise SystemExit(
            f"Sentence-Transformers source mismatch: {source_head} != {SOURCE_COMMIT}"
        )
    missing = [name for name in ADAPTER_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing holdout files: " + ", ".join(missing))


def write_manifest(attempts: list[str], batch: str) -> Path:
    from agent.config import LLM_MODEL, LLM_THINKING

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    payload = {
        "started_at_utc": timestamp,
        "experiment": "held_out_repository_generalization_n5",
        "batch": batch,
        "excluded_pilot_manifest": "manifest_20260807T103345Z.json",
        "excluded_pilot_reason": (
            "Oracle adapter false-rejected valid CLI output-path code because "
            "the public protocol exposed predictions.json as a literal code marker"
        ),
        "frozen_agent_commit": FROZEN_AGENT_COMMIT,
        "current_head": current_head,
        "frozen_agent_paths": list(FROZEN_AGENT_PATHS),
        "agent_paths_unchanged": True,
        "source_repo": "https://github.com/UKPLab/sentence-transformers",
        "source_version": "v5.7.0",
        "source_commit": SOURCE_COMMIT,
        "dataset": "sentence-transformers/stsb:test",
        "dataset_revision": DATASET_REVISION,
        "model": "sentence-transformers/all-mpnet-base-v2",
        "model_revision": MODEL_REVISION,
        "llm_model": LLM_MODEL,
        "llm_thinking": LLM_THINKING,
        "temperature": 0.0,
        "pipeline": "full",
        "max_eval_executions": 5,
        "adapter_sha256": {
            name: sha256(ROOT / name)
            for name in ADAPTER_FILES
        },
        "public_input_sha256": sha256(
            ROOT / "data" / "sentence_transformers_stsb" / "test_pairs.jsonl"
        ),
        "attempts": attempts,
    }
    path = LOG_DIR / f"manifest_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def result_path(attempt: str) -> Path:
    return ROOT / "evals" / "runs" / f"sentence_transformers_stsb_{attempt}" / "result.json"


def run_one(attempt: str, force: bool) -> int:
    result = result_path(attempt)
    log_path = LOG_DIR / f"{attempt}.log"
    if result.exists() and not force:
        print(f"[skip] {attempt} -> {result.relative_to(ROOT)}", flush=True)
        return 0
    env = os.environ.copy()
    env["PIPELINE"] = "full"
    env["SENTENCE_TRANSFORMERS_ATTEMPT"] = attempt
    command = [sys.executable, "run_sentence_transformers_stsb.py"]
    print(f"[run] {attempt}", flush=True)
    started = time.time()
    with log_path.open("w") as log:
        log.write(f"# {' '.join(command)}\n")
        log.write(f"# PIPELINE=full SENTENCE_TRANSFORMERS_ATTEMPT={attempt}\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(
        f"[done] {attempt} exit={process.returncode} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return process.returncode


@contextmanager
def run_lock():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"holdout runner already active: {LOCK_PATH}") from exc
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"pid={os.getpid()}\nstarted={time.time()}\n")
    try:
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", default="holdout_v2_n5")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    attempts = [f"{args.batch}_s{index}" for index in range(1, args.repeats + 1)]
    with run_lock():
        check_frozen_agent()
        manifest = write_manifest(attempts, args.batch)
        print(f"manifest: {manifest.relative_to(ROOT)}", flush=True)
        failures = 0
        for index, attempt in enumerate(attempts, start=1):
            print(f"\n[{index}/{len(attempts)}]", flush=True)
            code = run_one(attempt, args.force)
            if code:
                failures += 1
                if args.stop_on_fail:
                    break
        print(f"\nfinished runs={len(attempts)} process_failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
