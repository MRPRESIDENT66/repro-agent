#!/usr/bin/env python3
"""CPU-safe evaluation wrapper for MMPretrain ResNet-18 on CIFAR-10.

Runs the repository's own tools/test.py as a subprocess, parses the
accuracy/top1 metric from its output, and prints a strict JSON REPRO_RESULT line.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Paths relative to the repository root (assumed to be the working directory)
    repo_root = Path.cwd()
    test_script = repo_root / "tools" / "test.py"
    config_path = repo_root / "configs" / "resnet" / "resnet18_8xb16_cifar10.py"
    checkpoint_path = repo_root / "ckpt.pth"

    # Verify required files exist
    for path, desc in [
        (test_script, "test script"),
        (config_path, "config file"),
        (checkpoint_path, "checkpoint file"),
    ]:
        if not path.exists():
            print(f"Error: {desc} not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Build the command: python tools/test.py <config> <checkpoint>
    cmd = [
        sys.executable,
        str(test_script),
        str(config_path),
        str(checkpoint_path),
    ]

    # Run the evaluation as a subprocess, capturing stdout and stderr
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=3600,  # 1 hour timeout for CPU evaluation
        )
    except subprocess.TimeoutExpired:
        print("Error: evaluation timed out", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine stdout and stderr for parsing (mmengine may print to either)
    combined_output = result.stdout + "\n" + result.stderr

    # Parse the accuracy/top1 value from mmengine's output
    # Typical output line: "accuracy/top1: 94.82"
    pattern = r"accuracy/top1:\s*([\d.]+)"
    match = re.search(pattern, combined_output)

    if not match:
        print(
            "Error: could not find 'accuracy/top1' in evaluation output",
            file=sys.stderr,
        )
        print("Full stdout:", result.stdout, file=sys.stderr)
        print("Full stderr:", result.stderr, file=sys.stderr)
        sys.exit(1)

    top1_accuracy = float(match.group(1))

    # Print the required REPRO_RESULT line
    repro_result = {
        "metric": "top1_accuracy",
        "actual": top1_accuracy,
        "num_examples": 10000,
    }
    print(f"REPRO_RESULT {json.dumps(repro_result)}")


if __name__ == "__main__":
    main()
