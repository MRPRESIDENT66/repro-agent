# Repro-Agent

> An agent that reproduces a published ML result **blind**: hand it a model + a
> dataset + the claimed number, and it reads the repository, writes and runs the
> evaluation, and a deterministic verifier checks the result against a target the
> agent **never sees** — leaving a replayable audit trail.

Then it does the thing most "agent" projects skip: it **measures which parts of
the agent actually earn their cost.**

**→ Full write-up: [`evals/FINAL_REPORT.md`](evals/FINAL_REPORT.md)**

---

## The headline result: where does the multi-agent complexity pay off?

The agent is a pipeline — Navigator → Reproducer → Critic → *execute* → Reviewer
→ Repair. A natural question (and a fair interview question): is all that worth
it, or would one agent do? So I made the pipeline depth a switch and ran a
controlled ablation, N=5 each:

- **solo** — Reproducer only → execute (single-agent baseline)
- **team** — Navigator + Reproducer + Critic → execute (pre-execution collaboration)
- **full** — adds the Reviewer + Repair loop

| | solo | team | full |
|---|---|---|---|
| **easy task** (DistilBERT SST-2) | 5/5 | 5/5 | 5/5 |
| **hard task** (OpenOOD EBO) | **0/5** | **0/5** | **3/5** |
| cost / run (hard) | ¥0.036 | ¥0.122 | ¥0.262 |

**The finding (counterintuitive):** on the easy task the extra agents buy
**nothing** — solo already passes, at 1/6 the cost. On the hard task,
pre-execution collaboration also buys **nothing** (team still 0/5); success
appears **only** when the post-execution **repair loop** is switched on (0 → 3/5).

So the value of the multi-agent design is concentrated in **one mechanism — the
"run it, read the real error, fix, re-run" repair loop — and only on tasks that
fail first.** Adding more agents to *discuss before running* doesn't move the
needle. (Implication: the pipeline can be simplified to Reproducer + repair loop.)

---

## It works, and it generalizes (N=5, blind)

Same agent, no orchestration changes per task — a new task is ~200 lines of
config + a 9-line runner.

| Reproduction task | type / domain | backend | target | passed |
|---|---|---|---|---|
| DistilBERT SST-2 | NLP sentiment | subprocess | 91.06 acc | **5/5** |
| mmpretrain ResNet-18 | image cls — clone & navigate (mmcv) | Docker | 94.82 top-1 | **5/5** |
| detectors ResNet-18 CIFAR-100 | image cls — timm registration | subprocess | 79.26 top-1 | **4/5** |
| OpenOOD EBO | OOD detection (composite AUROC) | Docker | 87.58 AUROC | **3/5** |
| RobustBench Carmon2019 | adversarial robustness (AutoAttack) | subprocess | 52.0 robust acc | **5/5** |
| **total** | 4 task-types · 2 backends | | | **22/25** |

Difficulty tracks the repair rate: easy tasks pass first-try; the hard ones fail
first and lean on the repair loop.

---

## Why a "pass" is trustworthy (blind + provenance gate)

The agent's prompt contains only the public task (`model + dataset + metric`);
the expected value and tolerance live in an external verifier. A run matches only
when a *successful* command prints structured evidence —

```text
REPRO_RESULT {"metric":"top1_accuracy","actual":94.82,"num_examples":10000}
```

— **and** that evidence comes from a real evaluation: a script that loads a model
+ data and computes the metric, or one that delegates to the repo's own eval
entry against the checkpoint. A bare `echo`/`printf` of the number is rejected.
Each run writes `result.json`, `commands.sh`, and per-role transcripts for replay.

Tightening this gate repeatedly caught real problems — a `0.91` vs `91.0` unit
ambiguity, an echo relay, and (during the final study) two false-negatives on
wrapper-delegated and library-API evals, both fixed (see the report's appendix).

---

## Supporting ablation: retrieval for large-repo navigation

Can retrieval find the eval entry + config in a **1858-file** repo (`mmpretrain`)?
recall@5, hint-light queries:

| keyword | BM25 | dense (embeddings) | hybrid | **+ LLM rerank** |
|---|---|---|---|---|
| 60% | 60% | 50% | 60% | **80%** |

**Finding:** the retrieval *algorithm* barely matters (names are literal in
paths; dense does **not** beat BM25). The +20pp win is the **LLM reranker**
disambiguating the true entry (`tools/test.py`) from look-alikes.

---

## How it works

```
reproduction task (model + dataset + claimed metric)
        │
        ▼
  Navigator → Reproducer → Critic → [ execute eval ] → Reviewer → Repair ─┐
  (each role = an isolated LLM context, generating its own search_repo      │
   queries at runtime; a fixed loop + a deterministic contract decide       │
   when to repair and when to stop)                                         │
        │                              └──────── repair loop, ≤4 rounds ◄────┘
        ▼
  deterministic verify (plain code, not an LLM): structured evidence +
  provenance, compared to the private target within tolerance
```

- **Execution:** a persistent subprocess session (fast, local venv) or a sandboxed
  `linux/amd64` Docker container when the env can't be built on the host
  (mmpretrain's mmcv) or strong isolation / a true offline cut is needed (OpenOOD).
- **The repair loop is a fixed control-flow skeleton wrapped around an
  LLM-driven diagnose-and-fix agent:** the *when* (a deterministic contract
  check; ≤4 rounds) is hard-coded; the *what* (what broke, how to fix it) is the
  model's call, fed the real execution error each round.
- **Cost accounting** is built in (tokens, peak context, ¥ per run).

---

## Repo layout

```
agent/multi_rag.py     the multi-agent + RAG + repair orchestrator (generic skeleton)
agent/loop.py          single-agent ReAct loop, function calling, cost accounting
evals/oracles/         per-task config: prompts · contract · provisioning (5 tasks)
evals/runs/            per-run artifacts (result.json, transcripts) — the audit trail
evals/FINAL_REPORT.md  the consolidated report — start here
exec/                  persistent subprocess + Docker session backends
verify/                blind, deterministic metric extraction + the provenance gate
retrieval/             the large-repo navigation retrieval ladder
run_*_multi_rag.py     one runner per task (PIPELINE=solo|team|full for the ablation)
app.py · serve_mcp.py  Gradio demo · MCP server (both use run_repro.py)
legacy/                first-phase single-agent + M1–M5 ablation runners (archived)
```

## Setup & run

Python 3.12; an OpenAI-compatible chat key (DeepSeek) + a DashScope key for
embeddings, in a gitignored `.env`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env:  LLM_API_KEY=…  LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-chat
#        DASHSCOPE_API_KEY=…  EMBEDDING_MODEL=text-embedding-v4

# reproduce one task, blind (full pipeline)
python run_distilbert_multi_rag.py

# the pipeline ablation: same task, fewer roles
PIPELINE=solo python run_openood_multi_rag.py    # Reproducer only
PIPELINE=team python run_openood_multi_rag.py    # + Navigator + Critic, no repair

pytest -q                                         # 98 tests
```

The Docker-backed tasks (mmpretrain, OpenOOD) run inside pre-provisioned images;
the irreducible env-hell (mmcv) is solved once in the image, not by the agent.

## Honest caveats

- **Small N** (5). Pass rates are indicative, not significance-tested; tasks are
  development tasks, **not held-out**.
- The two `detectors` tasks come from the same library and were added to exercise
  the repair loop, not for paper breadth.
- **mmpretrain blindness is soft** — its 94.82 is in the public repo's own
  model-zoo metafile; "blind" means the task/verifier never reveal it, and the
  agent must still run the real `tools/test.py`.
- The subprocess backend is fast but **not** a security boundary; a strict
  held-out run should use the Docker backend and expose only the public task.
- `RepDistiller` is `artifact_blocked` (dead checkpoint host) — reported, not
  counted as a failure.
