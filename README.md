# Repro Agent

> An agent that takes an unfamiliar ML artifact (a model + a dataset + the
> published metric) and autonomously **sets up an environment, runs the public
> evaluation, verifies the result deterministically, and emits replayable
> evidence** — then **honestly measures how far it gets, and where it breaks.**

It does **not** claim to reproduce whole papers. It runs *public, lightweight
eval results* (mostly "load the released checkpoint and run eval"), and the
point is the **honest, staged measurement** — not a magic reproducer.

---

## The question

> Can an agent, in a controlled environment, get a stranger's ML artifact to
> produce its published number — and leave an auditable trail?

Real artifacts are messy: dead dataset links, drifted APIs, non-obvious load
mechanisms, environments that won't build. So the interesting output isn't
pass/fail — it's **how far the agent gets** (7 stages) and **why it stops**.

## Blind verification protocol V1

The Agent task/prompt contains only the public task (`model + dataset + metric`);
the expected value and tolerance stay in the external verifier. A result counts
only when a successful command prints structured evidence:

```text
REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000}
```

The independent verifier checks that evidence against the private claim.
Assistant `FINAL` text, unstructured numbers, and direct `echo/printf` result
relays are ignored. A provenance gate also requires a generated eval script
that loads data, predicts, and emits the evidence. Each run writes
`result.json`, `commands.sh`, and `transcript.jsonl` for audit/replay.

## Results — blind verification, reliability (deepseek-chat)

Each oracle run **N=5**, every match gated by the blind protocol above
(structured evidence from a real eval command; `num_examples` = full set;
provenance). The agent never sees the target value (development tasks, **not
held-out**).

| Oracle | Domain | Difficulty | Reproduced | avg steps |
|---|---|---|---|---|
| `cifar10_resnet20` (92.60) | vision | easy — torch.hub | **5/5 = 100%** | 4.8 |
| `distilbert_sst2` (91.06) | NLP | medium — transformers | **5/5 = 100%** | 5.8 |
| `resnet18_cifar100` (79.26) | vision | hard — registration helper + timm | **4/5 = 80%** | 9.6 |
| `mmpretrain_resnet18_cifar10` (94.82) | vision | **clone-and-navigate** — 1858-file repo, Docker | **2/3 = 67%** (exact) | 10–11 |

All matched runs reproduced the published number **exactly**. The blind rates
**match the earlier non-blind baseline** — the agent reproduces just as reliably
*without seeing the target*, so it was never relying on knowing the answer.
**Difficulty, not luck, is the variable:** the hard oracle once failed (1/5,
rabbit-holing into manual architecture reconstruction); the rest land it via a
*transferable* fix — read the model-card prose; "not registered" → install+import
the helper.

The fourth oracle is the **main act**: not a library load but a true
**clone → navigate → run → verify** on mmpretrain (**2/3 blind → 94.82 exact**;
the 1/3 failure rabbit-holes offline into `datasets.load_dataset`). The agent
uses `search_repo` to find `tools/test.py` + the cifar10 config in a 1858-file
repo, then runs the repo's **own** eval harness (the env-block — `mmcv` — is
pre-provisioned in the `repro-mmpretrain` Docker image; see `evals/RESULTS.md`),
all behind a two-phase network. It also surfaced two honest verification
findings: a provenance-gate blind spot for *delegation* (fixed + re-verified
offline), and a clean demonstration that the gate **rejects an `echo` of the
right number** and forces a real eval.

### Retrieval ladder for large-repo navigation

Can retrieval locate the eval entry + config in a **1858-file** repo
(`mmpretrain`)? recall@5, 5 hint-light queries:

| keyword | BM25 | dense (embeddings) | hybrid | **+ LLM rerank** |
|---|---|---|---|---|
| 60% | 60% | 50% | 60% | **80%** |

**Honest finding:** the retrieval algorithm barely matters here (model/dataset
names are literal in file paths; dense embeddings do **not** beat BM25). The
+20pp win is the **LLM reranker** disambiguating the true entry script
(`tools/test.py`) from look-alikes (`slurm_test.sh`, …). Retrieval recalls
candidates; reasoning picks the entry.

## How it works

```
manifest (model + dataset + claim)
      │
      ▼
  persistent session  ──►  ReAct loop over shell/file actions
  (workdir + venv,           ├─ read the model card / navigate the repo
   secret-scrubbed,          ├─ set up env, write + run the eval
   recorded)                 └─ tiered self-repair on error (env hell)
      │
      ▼
  parse metric  ──►  deterministic verify (code, not LLM)  ──►  replayable evidence
                     vs. published value within tolerance
```

- **Execution** is a persistent subprocess session (state persists across
  steps; secrets scrubbed). LLM = DeepSeek; embeddings = DashScope.
- **Transport** is native function calling (`bash` / `search_repo` / `finish`
  tool schemas). The original text protocol (regex-parsed ```bash blocks) is
  kept behind `--no-fc` as the ablation twin — "tool calls vs text parsing" is
  a measurable comparison, not a fashion choice.
- **Cost accounting** is built in: each run's `result.json` reports tokens
  (prompt / cached / completion), peak context in real tokens, and yuan cost
  (prices configurable via `.env`).
- **Verification** is blind and deterministic: only structured stdout from a
  successful evaluation with provenance is accepted; the private comparison is
  plain code.
- **Evaluation** reports staged pass rates (`repo_inspected → … → claim_matched`)
  + a failure taxonomy; `eligibility` vs `outcome` are separated so a task can't
  be moved out of the denominator after the agent fails.

## Repo layout

```
agent/       LLM (function calling + cost accounting) + ReAct loop + self-repair
agents/      multi-agent Lead/Reproducer/Verifier (isolation + concurrency ablations)
exec/        persistent session (shell/file actions, replay log)
verify/      deterministic metric extraction + comparison
retrieval/   navigation ladder: corpus · keyword/BM25/dense/hybrid/rerank
evals/       benchmark manifests · significance (clustered bootstrap) · RESULTS.md
serve_mcp.py  MCP server: verify_evidence_line · navigate_repo · reproduce_artifact
run_repro.py · run_reliability.py · run_multiagent.py
```

## Setup & run

Python 3.12, an OpenAI-compatible chat key (DeepSeek) + DashScope key for
embeddings, in a gitignored `.env`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env:  LLM_API_KEY=…  LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-chat
#        DASHSCOPE_API_KEY=…  DASHSCOPE_BASE_URL=…  EMBEDDING_MODEL=text-embedding-v4

python run_repro.py evals/benchmark/cifar10_resnet20.yaml   # reproduce one oracle (function calling)
python run_repro.py evals/benchmark/cifar10_resnet20.yaml --no-fc   # text-protocol ablation twin
python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5 --workers 3  # N trials, concurrent
python run_multiagent.py                                    # isolation + parallel-vs-serial ablation
python -m retrieval.eval_nav --dense                        # the navigation ladder
python serve_mcp.py                                         # MCP server (stdio) for any MCP client
pytest -q
```

**The clone-and-navigate oracle (mmpretrain)** runs inside a *pre-provisioned*
Docker image — the agent navigates + runs the eval; the `mmcv` env-block is
solved once in the image (it is **not** the agent's job; see `docs/DESIGN.md`).
Build it once, then run:

```bash
docker build --platform linux/amd64 -f docker/mmpretrain.Dockerfile -t repro-mmpretrain:latest .
python run_repro.py evals/benchmark/mmpretrain_resnet18_cifar10.yaml   # blind clone→navigate→run→verify
```

## Honest status & caveats

- **Reproducibility crisis, as data:** `RepDistiller` = `artifact_blocked` (dead
  checkpoint host) — reported, not counted as agent failure. `mmpretrain` was
  `env_blocked` (mmcv won't build on Py3.12) until the Docker amd64 +
  prebuilt-wheel recipe **unblocked it** — now a real clone-and-navigate oracle.
- Three oracles are **library-load**; the fourth (`mmpretrain`) is a true
  **clone-and-navigate-and-run** (Docker backend, two-phase network). The
  retrieval ladder is still measured on the cloned source for the algorithm
  comparison.
- Small n (5–8 runs/oracle), single model — effect sizes + staged rates, not
  significance claims.
- The default execution is a subprocess session (fast, MPS); for **untrusted**
  repos a pluggable `exec/docker_session.py` backend adds container isolation,
  resource caps, and a **two-phase network** (provision online → `go_offline()` →
  execution offline). The benchmark above ran on the subprocess backend (reviewed
  repos). Don't point the subprocess backend at code you don't trust.
- Blind V1 removes the expected value from the Agent prompt and clears stale
  run artifacts, but subprocess execution is not a filesystem security boundary.
  A strict held-out run should use Docker and expose only the public task.
