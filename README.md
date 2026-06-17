# Repro-Agent

A blind multi-agent runtime for reproducing ML repository results: agents inspect a repo, write and execute an evaluation program, repair from real failures, and pass only when an independent verifier recomputes the metric from per-sample outputs.

![Architecture: blind inputs feed a generic role pipeline that emits per-sample predictions, which an independent verifier recomputes against pinned gold labels.](docs/architecture.svg)

## Pipeline

```text
public task + repo + output contract
        │
        ▼
Navigator ──handoff──▶ Reproducer ──program──▶ Critic
        │                              │          │
        └──── repo-navigation RAG ◀────┴──────────┘
                                      │
                                      ▼
                              execute program
                                      │
                    stdout/stderr + public verifier diagnostics
                                      │
                                      ▼
                              Reviewer / Repair loop
                                      │
                                      ▼
                         per-sample artifact + result.json
                                      │
                                      ▼
                         fail-closed metric recomputation
```

| Role | Context boundary | Responsibility |
|---|---|---|
| Navigator | public task, repo snippets, retrieved evidence | Find entry points, assets, metric semantics, and unresolved risks. |
| Reproducer | public task, navigator handoff, retrieved source | Write the complete evaluation script and output per-sample predictions. |
| Critic | generated code, source evidence | Audit code before execution without seeing verifier gold or target values. |
| Reviewer | code, execution log, public verifier diagnostics | Decide whether the failure is contract, runtime, semantic, or workflow related. |
| Repair | previous code, failure summary, selected evidence | Patch the existing script, avoiding blind regeneration and duplicate retries. |

Each role starts from a fresh LLM context instead of inheriting the full chat history. Retrieval is repo-navigation oriented: BM25 lexical search, path/symbol signals, source snippets, optional LLM rerank, and dynamic queries generated from the current uncertainty, code, and failure logs.

## Failure-Grounded Repair

The repair loop is driven by a failure classifier over stdout/stderr and verifier diagnostics. It produces a compact failure summary, distinguishes contract/runtime/semantic/workflow failures, and can suggest restricted runtime probes for import, signature, path, or CLI uncertainty.

Runtime probes are soft hints, not mandatory gates: repairs may skip probing when source evidence is sufficient. The default repair policy is patch-first, with full-file replacement only as a fallback.

## Verifier

The agent never sees the hidden target metric. It must write a public artifact with per-sample predictions; the verifier loads pinned gold labels and recomputes the metric independently.

Fail-closed cases include missing artifact, malformed JSONL/CSV, wrong sample count, aggregate-only output, non-recomputable predictions, and values outside tolerance. Public diagnostics can be fed back to Reviewer/Repair, but hidden expected values are not exposed to the agent workspace.

## Experiment Results

Current summarized results live in [evals/RESULTS.md](evals/RESULTS.md). The
coverage table there is kept as an archived N=5 summary; the current ablation is
the simplified three-condition E2 (`solo`, `solo-repair`, `full`).

## Pipeline Ablation

All conditions use the same generic prompts, verifier, and execution budget; they differ only in orchestration depth and whether execution feedback is used.

| Condition | Roles | Execution feedback | Purpose |
|---|---|---|---|
| `solo` | Reproducer | no repair | Baseline one-shot code generation. |
| `solo-repair` | Reproducer + Repair | real logs + diagnostics | Isolate execution-grounded repair. |
| `full` | Navigator + Reproducer + Critic + Reviewer + Repair | real logs + diagnostics | Configurable collaboration mode, not assumed to be always best. |

N=5 results show that full collaboration is not universally strongest: repair feedback helps harder tasks, while extra handoffs can introduce workflow failures. Detailed tables, OpenOOD notes, cost, pass@k, and failure analysis are in [evals/RESULTS.md](evals/RESULTS.md).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=...
export LLM_MODEL=...
```

```bash
python run_distilbert_multi_rag.py
PIPELINE=solo-repair python run_openood_multi_rag.py
PIPELINE=full python run_robustbench_multi_rag.py
pytest -q tests --ignore=workspaces --ignore=repos
```

Useful paths:

- `agent/multi_rag.py` — top-level orchestration and execution/repair loop.
- `agent/types.py` — shared task/runtime configuration types.
- `agent/repair.py` — patch-first repair and repair validation.
- `agent/diagnostics.py` — generic public-contract diagnostics.
- `agent/runtime_probe.py` — restricted import/signature/path/CLI probes.
- `agent/generic_prompts.py` — task-agnostic role prompts.
- `agent/failure.py` — failure classification and probe suggestions.
- `retrieval/` — repo-navigation search and snippet extraction.
- `exec/` — subprocess/Docker execution sessions.
- `verify/` — fail-closed metric recomputation.
- `evals/oracles/` — public task specs and verifier contracts.

## Scope / Limitations

This is not a claim of zero-configuration reproduction for arbitrary repositories. Each task still needs a public task spec, execution command, sample contract, and hidden verifier assets.

This is an experiment-integrity runtime for cooperative agents, not a complete security sandbox against malicious code. The verifier rejects unverifiable outputs and target leakage in the artifact path, but it does not prove the workspace is adversary-proof.

Run artifacts under `evals/runs/`, `logs/`, `workspaces/`, and `repos/` are generated outputs. They are intentionally kept out of the main project narrative; only summarized, auditable results should be committed or documented in `evals/RESULTS.md`.
