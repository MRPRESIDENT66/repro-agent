#!/usr/bin/env python3
"""
CPU-safe evaluation wrapper for MMPretrain ResNet-18 CIFAR-10.
Runs the repository's own test.py as a subprocess, parses the top-1 accuracy
from its output, and prints a strict-JSON REPRO_RESULT line.
"""

import json
import os
import re
import subprocess
import sys

# The repository is in a subdirectory 'mmpretrain' relative to this script
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmpretrain")
TEST_SCRIPT = os.path.join(REPO_ROOT, "tools", "test.py")
CONFIG_PATH = os.path.join(
    REPO_ROOT, "configs", "resnet", "resnet18_8xb16_cifar10.py"
)
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "ckpt.pth")

# Validate paths exist
for path, label in [(TEST_SCRIPT, "test.py"), (CONFIG_PATH, "config")]:
    if not os.path.exists(path):
        print(f"Error: {label} not found at {path}", file=sys.stderr)
        sys.exit(1)

# Checkpoint must already exist locally (no internet available)
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: checkpoint not found at {CHECKPOINT_PATH}", file=sys.stderr)
    sys.exit(1)

# Build the command: python tools/test.py <config> <checkpoint>
cmd = [
    sys.executable,
    TEST_SCRIPT,
    CONFIG_PATH,
    CHECKPOINT_PATH,
]

# Run the evaluation as a subprocess, capturing stdout and stderr
try:
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout for CPU evaluation
    )
except subprocess.TimeoutExpired:
    print("Error: evaluation timed out", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error running evaluation: {e}", file=sys.stderr)
    sys.exit(1)

# Combine stdout and stderr for parsing
output = result.stdout + "\n" + result.stderr

# Parse the top-1 accuracy from mmengine's output format:
# Example: "accuracy/top1: 85.23"
# The metric key is "accuracy/top1" followed by a colon and the value.
pattern = r"accuracy/top1:\s*([\d.]+)"
match = re.search(pattern, output)

if not match:
    print("Error: could not parse accuracy/top1 from output", file=sys.stderr)
    print("Full output:", output, file=sys.stderr)
    sys.exit(1)

# Parse the accuracy value as a float (already in percentage points)
top1_accuracy = float(match.group(1))

# Validate the parsed value is in a reasonable range (0-100)
if not (0 <= top1_accuracy <= 100):
    print(f"Error: parsed accuracy {top1_accuracy} is out of range [0, 100]", file=sys.stderr)
    sys.exit(1)

# Print the required REPRO_RESULT line with strict JSON
repro_result = {
    "metric": "top1_accuracy",
    "actual": top1_accuracy,
    "num_examples": 10000
}
print(f"REPRO_RESULT {json.dumps(repro_result, separators=(',', ':'))}")
