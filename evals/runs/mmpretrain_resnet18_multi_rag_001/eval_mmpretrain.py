#!/usr/bin/env python3
"""
CPU-safe evaluation wrapper for MMPretrain ResNet-18 CIFAR-10.
Runs the repository's own test.py as a subprocess, parses the top-1 accuracy,
and prints a single REPRO_RESULT JSON line.
"""

import json
import os
import re
import subprocess
import sys

# Paths relative to the repository root (mmpretrain/)
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SCRIPT = os.path.join(REPO_DIR, "tools", "test.py")
CONFIG_PATH = os.path.join(
    REPO_DIR, "configs", "resnet", "resnet18_8xb16_cifar10.py"
)
CHECKPOINT_PATH = os.path.join(REPO_DIR, "ckpt.pth")

# Ensure the checkpoint exists
if not os.path.isfile(CHECKPOINT_PATH):
    print(f"Error: checkpoint not found at {CHECKPOINT_PATH}", file=sys.stderr)
    sys.exit(1)

# Build the command
cmd = [
    sys.executable,
    TEST_SCRIPT,
    CONFIG_PATH,
    CHECKPOINT_PATH,
]

# Run the evaluation as a subprocess, capturing stdout+stderr
try:
    proc = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=600,  # generous timeout for CPU-only
    )
except subprocess.TimeoutExpired:
    print("Error: evaluation timed out", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error running subprocess: {e}", file=sys.stderr)
    sys.exit(1)

# Combine stdout and stderr for parsing
output = proc.stdout + "\n" + proc.stderr

# Parse the top-1 accuracy from mmengine's output
# Expected format: "accuracy/top1: 94.82"
match = re.search(r"accuracy/top1:\s*([\d.]+)", output)
if not match:
    print("Error: could not parse accuracy/top1 from output", file=sys.stderr)
    print("Full output:", output, file=sys.stderr)
    sys.exit(1)

top1_accuracy = float(match.group(1))

# Print the required REPRO_RESULT line
result = {
    "metric": "top1_accuracy",
    "actual": top1_accuracy,
    "num_examples": 10000,
}
print(f"REPRO_RESULT {json.dumps(result)}")
