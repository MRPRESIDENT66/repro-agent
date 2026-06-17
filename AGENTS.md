# Codex Context

This repo is a blind multi-agent runtime for ML result reproduction. Agents inspect a target repo, write an evaluation script, execute it, repair from real failures, and pass only when an independent verifier recomputes metrics from per-sample artifacts.

## Read First

- `README.md` — short project overview, pipeline, verifier, run commands.
- `evals/RESULTS.md` — summarized experiments, ablations, failure analysis.
- `agent/multi_rag.py` — orchestration, role handoffs, execution/repair loop.
- `agent/failure.py` — failure classification and runtime-probe suggestions.
- `agent/generic_prompts.py` — task-agnostic role prompts.
- `retrieval/` — repo-navigation search and snippet extraction.
- `verify/` — fail-closed metric recomputation.
- `tests/` — regression tests for prompts, orchestration, verifier, failure logic.

## Do Not Scan By Default

- `evals/runs/` — generated run artifacts.
- `logs/` — generated logs.
- `workspaces/` — scratch task workspaces.
- `repos/` — cloned/source dependency snapshots.
- `.venv/`, `.venv-*/`, `data/`, `checkpoints/` — local environments/assets.

These paths are intentionally ignored by `.gitignore` and `.codexignore`. Summarized evidence should live in `evals/RESULTS.md`, not raw run directories.

## Common Commands

- `pytest -q tests --ignore=workspaces --ignore=repos`
- `python run_distilbert_multi_rag.py`
- `PIPELINE=solo-repair python run_openood_multi_rag.py`

## Design Constraints

- Keep the agent blind to hidden expected values and target metrics.
- Keep verifier logic fail-closed and based on recomputable per-sample outputs.
- Treat runtime probes as failure-classifier suggested soft hints, not mandatory gates.
- Keep `full` as a configurable collaboration mode, not a claim of always-best performance.
- Prefer patch-first repair over full-file regeneration.
