#!/usr/bin/env python3
"""CPU-safe wrapper to evaluate a ResNet-18 CIFAR-10 checkpoint using MMPreTrain's tools/test.py.

Usage:
    python eval_mmpretrain.py

Requires:
    - The repository's tools/test.py entry point
    - Config at configs/resnet/resnet18_8xb16_cifar10.py
    - Checkpoint at ckpt.pth
    - Data at data/cifar10/ (already present, offline)
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    # Paths relative to the repository root (assumed to be current working directory)
    repo_root = Path.cwd()
    config_path = repo_root / "configs" / "resnet" / "resnet18_8xb16_cifar10.py"
    checkpoint_path = repo_root / "ckpt.pth"

    # Validate paths exist
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Build the command: python tools/test.py <config> <checkpoint>
    # No --launcher needed (defaults to 'none'), CPU-only environment
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "test.py"),
        str(config_path),
        str(checkpoint_path),
    ]

    # Run the evaluation as a subprocess, capturing stdout+stderr
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=600,  # generous timeout for CPU evaluation
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation timed out", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Subprocess failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine stdout and stderr for parsing (mmengine may print to either)
    combined_output = result.stdout + "\n" + result.stderr

    # Parse the top-1 accuracy from mmengine's output format:
    # "accuracy/top1: 94.5670" (or similar)
    pattern = r"accuracy/top1:\s*([0-9]+\.[0-9]+)"
    match = re.search(pattern, combined_output)

    if not match:
        print("ERROR: Could not parse accuracy/top1 from output", file=sys.stderr)
        print("Full output:", combined_output, file=sys.stderr)
        sys.exit(1)

    top1_accuracy = float(match.group(1))

    # Print the required REPRO_RESULT line with the parsed value
    # num_examples is fixed at 10000 (CIFAR-10 test set size)
    repro_result = {
        "metric": "top1_accuracy",
        "actual": top1_accuracy,
        "num_examples": 10000,
    }
    print(f"REPRO_RESULT {json.dumps(repro_result)}")


if __name__ == "__main__":
    main()
