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
**budget-matched** ablation — five conditions sharing one execution budget
(≤5 evals, so "more attempts" is held constant), N=5 each:

| condition (≤5 evals each) | easy (DistilBERT) | hard (OpenOOD) |
|---|---|---|
| solo — Reproducer only | 5/5 | 0/5 |
| team — + Navigator + Critic (no repair) | 5/5 | 0/5 |
| solo-retry — re-generate, no error feedback | 5/5 | **0/5** |
| solo-repair — single agent + feedback repair | 5/5 | **0/5** |
| **full** — Navigator + Critic + Reviewer + Repair | 5/5 | **3/5** |

**The finding:** on the easy task every condition passes — the machinery is
unnecessary. On the hard task, **no reduced condition produces any success** at
the shared budget: not more attempts (solo-retry 0/5), not single-agent
feedback-repair (solo-repair 0/5), not pre-execution collaboration (team 0/5).
**Only the full pipeline reaches 3/5.** So the multi-agent components are
**complementary on hard tasks, not redundant** — success needs the pre-execution
grounding *and* the reviewer-guided repair loop together.

> This **overturned my own earlier (premature) claim.** A first cut compared only
> solo/team/full, where solo ran *once* and full ran up to *five* times; that made
> it look like "the repair loop is everything, drop the Critic." The
> budget-matched controls show feedback-repair *alone* is 0/5 — the pipeline can't
> be reduced to it. (Surfaced by an external review; see `evals/FINAL_REPORT.md`,
> Appendix C.)

---

## It runs across domains (N=5, blind)

Same agent, no orchestration changes per task — a new task is ~200 lines of
config + a 9-line runner. Tables are generated from the run artifacts by
`python evals/report_tables.py`. `blind`: *strict* = target absent from the
workspace; *soft* = in the public repo but never surfaced by task/verifier.

| Reproduction task | type / domain | backend | target | blind | passed |
|---|---|---|---|---|---|
| DistilBERT SST-2 | NLP sentiment | subprocess | 91.06 acc | strict | **5/5** |
| mmpretrain ResNet-18 | image cls — clone & navigate (mmcv) | Docker | 94.82 top-1 | soft | **5/5** |
| detectors ResNet-18 CIFAR-100 | image cls — timm registration | subprocess | 79.26 top-1 | strict | **4/5** |
| OpenOOD EBO | OOD detection (composite AUROC) | Docker | 87.58 AUROC | strict | **3/5** |
| RobustBench Carmon2019 | adversarial robustness (AutoAttack) | subprocess | 52.0 robust acc | strict | **4/5** |
| **total** | 4 task-types · 2 backends | | | | **21/25** |

Difficulty tracks the repair rate: easy tasks pass first-try; the hard ones fail
first and need the full pipeline. (All five are **development** tasks — no
held-out split; see caveats.)

---

## What the verifier checks (and its real limit)

The expected value and tolerance live in the external verifier, not in the
runtime task. Under the V2 protocol, the generated evaluation writes per-sample
predictions or scores to `predictions.json`; the verifier ignores any metric the
agent prints and independently recomputes the metric against pinned reference
labels or the fixed evaluation semantics.

This prevents accidental passes from a guessed, echoed, or hardcoded aggregate
number: without a complete per-sample output file, verification fails closed.
All five reported task types use verifier-side recomputation.

> **Threat model:** this is an experiment-integrity check for a cooperative,
> non-adversarial Agent, **not a security sandbox**. It does not claim to stop an
> adaptive attacker that deliberately searches outside the workspace, reads
> verifier assets, or fabricates a full prediction vector engineered to match the
> hidden target. Docker-backed tasks provide stronger filesystem isolation; the
> subprocess backend does not.

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
run_*_multi_rag.py     one runner per task (PIPELINE=solo|team|solo-retry|solo-repair|full)
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

# the budget-matched pipeline ablation (PIPELINE ∈ solo|team|solo-retry|solo-repair|full)
PIPELINE=solo-retry  python run_openood_multi_rag.py   # re-generate, no feedback
PIPELINE=solo-repair python run_openood_multi_rag.py   # single agent + feedback repair

# shared role prompts + public-artifact-only validation/feedback
PROMPT_MODE=generic DISTILBERT_ATTEMPT=generic_001 python run_distilbert_multi_rag.py

python evals/report_tables.py                     # regenerate the E1/E2 tables from artifacts
pytest -q                                          # run the full local regression suite
```

The Docker-backed tasks (mmpretrain, OpenOOD) run inside pre-provisioned images;
the irreducible env-hell (mmcv) is solved once in the image, not by the agent.

## Honest caveats

- **Small N** (5), and the hard-task E2 result rests on **one** hard task
  (OpenOOD). Pass rates are indicative, not significance-tested. All five are
  **development** tasks (prompts iterated against them) — **no held-out split**,
  so "runs across domains" is scoped to this suite.
- **Oracle specialization is real in the reported runs:** the per-task prompts
  hand the agent APIs, field names, and known gotchas. `PROMPT_MODE=generic`
  replaces every role prompt with one shared prompt and disables specialized code
  validators and repair diagnostics. It exposes only the public task, the exact
  command used to invoke the generated program, and the public result-artifact
  schema required for scoring. Generic roles also share one restricted,
  budgeted `runtime_probe` tool for import smoke tests, Python signatures,
  workspace-relative path listings, and CLI help. Probe commands are recorded
  separately from verifier-visible evaluation commands in
  `runtime_probes.json`; this is an auditable non-malicious diagnostic guard,
  not a security sandbox. Execution-driven generic repairs must use one probe
  for concrete import/API/path failures, cannot re-enter package initializers
  already proven broken by a public traceback, and cannot resubmit unchanged
  code. Generic results are not yet included in the reported tables.
- **Verifier-side recomputation prevents non-adversarial result fabrication, not
  malicious attacks.** It rejects missing, malformed, wrong-count, or aggregate-only
  outputs, but the subprocess backend is not a sandbox and does not defend against
  an Agent deliberately trying to read verifier assets.
- **mmpretrain is soft-blind** (94.82 is in the repo's own model-zoo metafile);
  the other four are strict-blind (RobustBench's README `52.00%` leak is scrubbed
  at provisioning). Don't mix blind levels when summarizing — hence the column.
- The two `detectors` tasks come from the same library and were added to exercise
  the repair loop, not for paper breadth. `RepDistiller` is `artifact_blocked`
  (dead checkpoint host) — reported, not counted as a failure.
