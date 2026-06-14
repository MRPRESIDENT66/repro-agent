"""Run one strict-blind collaborative Multi-Agent + RAG OpenOOD experiment."""

import os

from agent.multi_rag import run_oracle
from evals.oracles.openood_ebo import make_config

if __name__ == "__main__":
    run_oracle(make_config(os.environ.get("OPENOOD_MULTI_RAG_ATTEMPT", "002")))
