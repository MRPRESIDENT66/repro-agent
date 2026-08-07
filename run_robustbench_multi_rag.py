"""Run one strict-blind collaborative Multi-Agent + RAG RobustBench experiment."""

import os

from agent.pipeline import run_oracle
from evals.catalog import make_config

if __name__ == "__main__":
    run_oracle(
        make_config("robustbench_carmon", os.environ.get("ROBUSTBENCH_ATTEMPT", "001"))
    )
