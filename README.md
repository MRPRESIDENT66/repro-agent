# Repro-Agent

[English](README.md) | [中文](README.zh-CN.md)

A **blind multi-agent runtime** for reproducing ML repository results. A team of
role-specialized LLM agents inspects a repo, writes and executes an evaluation
program with **native tool calling**, **self-corrects from real execution
failures**, and is graded by an **independent, fail-closed evaluation harness**
that recomputes the metric from per-sample outputs — the agent never sees the
target number.

Built from scratch (the agent loop, retrieval, repair loop, and evaluator are
hand-implemented on a provider-agnostic OpenAI-compatible API) to make the
internals of an agentic system explicit rather than hidden behind a framework.

![Architecture: blind inputs feed a generic role pipeline that emits per-sample predictions, which an independent verifier recomputes against pinned gold labels.](docs/architecture.svg)

## What this project demonstrates

| Capability | What it is here | Where |
|---|---|---|
| **Multi-agent orchestration** | A role-specialized state machine (Navigator → Reproducer → Critic → execute → Reviewer → Repair) with per-role context isolation | [`agent/pipeline.py`](agent/pipeline.py) |
| **Tool use / function calling** | Native OpenAI function-calling agent loop, sequential tool dispatch, context compression | [`agent/loop.py`](agent/loop.py) |
| **Self-correction (Reflexion-style)** | A failure-classified, execution-grounded repair loop; patch-first edits over blind regeneration | [`agent/repair.py`](agent/repair.py), [`agent/failure.py`](agent/failure.py) |
| **RAG / retrieval** | Repo-navigation retrieval: BM25 lexical search + path/symbol signals + LLM reranking + dynamic query rewriting | [`retrieval/`](retrieval/) |
| **LLM evaluation & guardrails** | Blind, fail-closed verifier that recomputes the metric from per-sample artifacts and rejects unverifiable / leaked outputs | [`verify/`](verify/) |
| **Sandboxed code execution** | Subprocess + Docker execution sessions with two-phase network isolation | [`exec/`](exec/) |
| **Observability** | Per-call token + cost accounting, full transcripts, and replayable command scripts | [`agent/llm.py`](agent/llm.py) |
| **Evaluation methodology** | Budget-fair ablation across orchestration depths, `pass@k`, mean cost, failure-mode breakdown | [`evals/`](evals/) |
| **Deterministic agent testing** | `ScriptedLLM` drives the whole control flow with no API/tokens for fast, reproducible tests | [`tests/`](tests/) |

Stack: Python, OpenAI-compatible function calling (provider-agnostic; runs on
DeepSeek/any OpenAI-style endpoint), BM25 retrieval, Docker, `pytest`.

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

**Context engineering:** each role starts from a fresh LLM context instead of inheriting the full chat history, which keeps prompts focused and bounds token growth. **Retrieval (RAG)** is repo-navigation oriented: BM25 lexical search, path/symbol signals, source snippets, optional LLM rerank, and dynamic query rewriting generated from the current uncertainty, code, and failure logs.

## Self-Correction: Failure-Grounded Repair Loop

The repair loop is a Reflexion-style self-correction mechanism driven by a failure classifier over stdout/stderr and verifier diagnostics. It produces a compact failure summary, distinguishes contract/runtime/semantic/workflow failures, and can suggest restricted runtime probes (a constrained tool-use surface) for import, signature, path, or CLI uncertainty.

Runtime probes are soft hints, not mandatory gates: repairs may skip probing when source evidence is sufficient. The default repair policy is patch-first, with full-file replacement only as a fallback.

## Evaluation Harness & Guardrails (the Verifier)

The agent never sees the hidden target metric. It must write a public artifact with per-sample predictions; the verifier loads pinned gold labels and recomputes the metric independently — an offline, fail-closed eval that cannot be passed by guessing or echoing the published number.

Fail-closed cases include missing artifact, malformed JSONL/CSV, wrong sample count, aggregate-only output, non-recomputable predictions, and values outside tolerance. Public diagnostics can be fed back to Reviewer/Repair, but hidden expected values are not exposed to the agent workspace.

## Observability

Every LLM call accumulates token usage and cost (with cache-hit accounting), so a
run's cost is a delta of two snapshots. Each run emits the full per-role
transcript, RAG/probe traces, and a replayable `commands.sh`, making any verdict
auditable and reproducible after the fact.

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

- `agent/pipeline.py` — top-level orchestration state machine and execution/repair loop.
- `agent/contracts.py` — public task context and generic code/report/review validators.
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
