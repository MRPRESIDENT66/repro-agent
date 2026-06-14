#!/usr/bin/env python3
"""CPU-safe wrapper for MMPreTrain evaluation on ResNet-18 CIFAR-10.

Runs tools/test.py as a subprocess, parses the printed top-1 accuracy,
and prints a strict-JSON REPRO_RESULT line.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Paths relative to the repository root (assumed to be CWD)
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

    # Build command: python tools/test.py <config> <checkpoint>
    # --launcher none is the default, but we pass it explicitly for clarity
    cmd = [
        sys.executable,
        str(test_script),
        str(config_path),
        str(checkpoint_path),
        "--launcher",
        "none",
    ]

    # Run the evaluation as a subprocess, capturing stdout and stderr
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=600,  # generous timeout for CPU-only evaluation
        )
    except subprocess.TimeoutExpired:
        print("Error: evaluation timed out", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running subprocess: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine stdout and stderr for parsing (mmengine may print to either)
    combined_output = result.stdout + "\n" + result.stderr

    # Parse the top-1 accuracy from mmengine's output.
    # Expected format: "accuracy/top1: 94.82" (or similar)
    # We look for the pattern after "accuracy/top1:" which may have whitespace
    pattern = r"accuracy/top1:\s*([\d.]+)"
    match = re.search(pattern, combined_output)

    if not match:
        # If parsing fails, print the captured output for debugging
        print("Error: could not parse top-1 accuracy from output", file=sys.stderr)
        print("--- stdout ---", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print("--- stderr ---", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    top1_accuracy = float(match.group(1))

    # Validate the parsed value is in a reasonable range (0-100)
    if not (0 <= top1_accuracy <= 100):
        print(
            f"Error: parsed accuracy {top1_accuracy} is out of range [0, 100]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print the required REPRO_RESULT line
    output = {
        "metric": "top1_accuracy",
        "actual": top1_accuracy,
        "num_examples": 10000,
    }
    print(f"REPRO_RESULT {json.dumps(output)}")


if __name__ == "__main__":
    main()
