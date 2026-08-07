"""Run the second frozen held-out task: OpenAI CLIP on CIFAR-10."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config


if __name__ == "__main__":
    run_oracle(
        make_config(
            "clip_vitb32_cifar10",
            os.environ.get("CLIP_HELDOUT_ATTEMPT", "001"),
        )
    )
