"""Run one strict-blind collaborative Multi-Agent + RAG mmpretrain CIFAR-10 experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config("mmpretrain_resnet18", os.environ.get("MMPRETRAIN_ATTEMPT", "001"))
    )
