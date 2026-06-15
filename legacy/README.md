# legacy/ — earlier single-agent & ablation experiments

Archived runners from the project's first phase (single-agent ReAct loop and the
M1–M5 ablations). They are **not** part of the current multi-agent + RAG pipeline
(`agent/multi_rag.py` + `evals/oracles/` + the `run_*_multi_rag.py` scripts at the
repo root), but they produced results cited in the report's appendix
(reliability N=5, the M5 multi-agent isolation/concurrency ablation, the M4
context-compression ablation).

Kept for reproducibility and history, moved out of the top level to keep the
current pipeline easy to find.

| script | what it ran |
|---|---|
| `run_repro.py` *(stays at repo root — shared library)* | single-agent reproduction of one benchmark task; `reproduce()`/`build_task()` are imported by `app.py` and `serve_mcp.py` |
| `run_reliability.py` | reliability harness — run a task N times (blind) |
| `run_multiagent.py` | M5 multi-agent isolation/concurrency ablation |
| `run_m4.py` | M4 context-compression ablation |
| `run_m1.py` | M1 end-to-end single-agent runner |
| `run_openood_blind.py` | single-agent strict-blind OpenOOD EBO |

**Running them:** each file inserts the repo root on `sys.path`, so run from the
repo root, e.g.

    python legacy/run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5
