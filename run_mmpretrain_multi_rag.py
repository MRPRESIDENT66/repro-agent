"""Run one strict-blind collaborative Multi-Agent + RAG mmpretrain CIFAR-10 experiment."""

import os

from agent.multi_rag import run_oracle
from evals.oracles.mmpretrain_resnet18 import make_config

if __name__ == "__main__":
    run_oracle(
        make_config(os.environ.get("MMPRETRAIN_ATTEMPT", "001")),
        pipeline=os.environ.get("PIPELINE", "full"),
        prompt_mode=os.environ.get("PROMPT_MODE", "specialized"),
    )
