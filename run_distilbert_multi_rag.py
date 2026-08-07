"""Run one strict-blind collaborative Multi-Agent + RAG DistilBERT SST-2 experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config("distilbert_sst2", os.environ.get("DISTILBERT_ATTEMPT", "001"))
    )
