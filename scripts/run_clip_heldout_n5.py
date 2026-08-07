"""Run the frozen second held-out task five times."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config


PREFIX = os.environ.get("CLIP_HELDOUT_PREFIX", "heldout2_n5")


if __name__ == "__main__":
    for seed in range(1, 6):
        run_oracle(make_config("clip_vitb32_cifar10", f"{PREFIX}_s{seed}"))
