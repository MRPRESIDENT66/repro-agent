#!/usr/bin/env python3
"""CPU-safe evaluation wrapper for mmpretrain ResNet-18 CIFAR-10.

Runs the repository's own tools/test.py as a subprocess, parses the
accuracy/top1 metric from its output, and prints a strict-JSON REPRO_RESULT line.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Paths relative to the repository root (assumed to be CWD)
    config_path = "configs/resnet/resnet18_8xb16_cifar10.py"
    checkpoint_path = "ckpt.pth"

    # Verify files exist before launching
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Build the command: use the same Python interpreter that runs this script
    cmd = [
        sys.executable,
        "tools/test.py",
        config_path,
        checkpoint_path,
    ]

    # Run the evaluation as a subprocess, capturing stdout and stderr
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # generous timeout for CPU evaluation
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation timed out after 600 seconds", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: Could not run tools/test.py: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine stdout and stderr for parsing (mmengine may print to either)
    combined_output = proc.stdout + "\n" + proc.stderr

    # Parse the top-1 accuracy from mmengine's output.
    # Typical line: "accuracy/top1: 93.45"
    pattern = r"accuracy/top1:\s*([0-9]+\.?[0-9]*)"
    match = re.search(pattern, combined_output)

    if not match:
        print("ERROR: Could not parse accuracy/top1 from evaluation output", file=sys.stderr)
        print("=== STDOUT ===", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print("=== STDERR ===", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)

    top1_accuracy = float(match.group(1))

    # Build the strict JSON result
    result = {
        "metric": "top1_accuracy",
        "actual": top1_accuracy,
        "num_examples": 10000,
    }

    # Print exactly one REPRO_RESULT line
    print(f"REPRO_RESULT {json.dumps(result, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
