"""Run one strict-blind collaborative Multi-Agent + RAG OpenOOD experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config("openood_ebo", os.environ.get("OPENOOD_MULTI_RAG_ATTEMPT", "002"))
    )
