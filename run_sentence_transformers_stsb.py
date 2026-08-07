"""Run one frozen-agent held-out Sentence-Transformers STS-B experiment."""

import os

from agent.pipeline import run_oracle
from evals.oracles.sentence_transformers_stsb import make_config


if __name__ == "__main__":
    run_oracle(
        make_config(os.environ.get("SENTENCE_TRANSFORMERS_ATTEMPT", "001")),
        pipeline=os.environ.get("PIPELINE", "full"),
    )
