"""Run one strict-blind Multi-Agent + RAG detectors/timm VGG16-bn CIFAR-10 experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config(
            "detectors_vgg16_cifar10",
            os.environ.get("DETECTORS_ATTEMPT", "001"),
        )
    )
