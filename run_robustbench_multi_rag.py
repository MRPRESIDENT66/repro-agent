"""Run one strict-blind collaborative Multi-Agent + RAG RobustBench experiment."""

import os

from agent.multi_rag import run_oracle
from evals.oracles.robustbench_carmon import make_config

if __name__ == "__main__":
    run_oracle(make_config(os.environ.get("ROBUSTBENCH_ATTEMPT", "001")))
