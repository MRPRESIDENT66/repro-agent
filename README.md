# Repro-Agent

Generic ML artifact reproduction agent.

The agent receives only a public reproduction task, a public execution command,
and a public result-artifact contract. It must inspect the provisioned workspace,
retrieve source context, generate an evaluation program, execute it, and repair
failures from public logs until the deterministic verifier can score the
artifact.

This branch keeps only the generic implementation path. Historical experiment
reports and run artifacts are intentionally excluded.

## What Is Included

- `agent/multi_rag.py`: generic multi-role orchestration with dynamic RAG,
  execution, review, and feedback repair.
- `agent/generic_prompts.py`: shared task-agnostic role prompts.
- `evals/oracles/`: task configuration, provisioning hooks, public artifact
  contracts, and deterministic verifier-side recomputation.
- `verify/`: deterministic result extraction and provenance checks.
- `exec/`: persistent subprocess and Docker-backed execution sessions.
- `retrieval/`: repository indexing and search helpers.
- `run_*_multi_rag.py`: one runner per task. Use `PIPELINE` to choose the
  ablation condition.

## Pipeline Conditions

All conditions use the same generic prompts and public contract. They differ
only in orchestration depth and post-execution feedback:

- `solo`: Reproducer only, one execution.
- `team`: Navigator + Reproducer + Critic, one execution.
- `solo-retry`: Reproducer retries without execution feedback.
- `solo-repair`: Reproducer repairs using the public execution error.
- `full`: Navigator + Reproducer + Critic + Reviewer + repair loop.

The shared budget is one initial execution plus up to four follow-up executions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the LLM and embedding credentials in `.env`:

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
DASHSCOPE_API_KEY=...
EMBEDDING_MODEL=...
```

Some oracle tasks expect pre-provisioned local assets, cached datasets, or Docker
images. See the corresponding file in `evals/oracles/` for task-specific runtime
requirements.

## Run

```bash
python run_distilbert_multi_rag.py
PIPELINE=solo-repair python run_openood_multi_rag.py
PIPELINE=full python run_robustbench_multi_rag.py
```

Run the local regression suite:

```bash
pytest -q
```

## Scope

This is not a security sandbox and does not defend against an adversarial agent.
The verifier is an experiment-integrity check for cooperative runs: it rejects
missing, malformed, wrong-count, or aggregate-only outputs and recomputes metrics
from public result artifacts.
