"""Run one strict-blind Multi-Agent + RAG detectors/timm ResNet-18 CIFAR-100 experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config(
            "detectors_resnet18_cifar100",
            os.environ.get("DETECTORS_ATTEMPT", "001"),
        )
    )
