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

## Results (deepseek-chat, hint-light, anti-hardcode gated)

The agent is told only *what* to reproduce, never *how*. Every "match" requires
the eval script to actually load the data and predict over it (no hardcoded
print), spot-checked by re-running the agent's own script independently.

| Oracle | Domain | Difficulty | Reproduced | avg steps |
|---|---|---|---|---|
| `cifar10_resnet20` (92.60) | vision | easy — torch.hub | **8/8 = 100%** | 2.1 |
| `distilbert_sst2` (91.06) | NLP | medium — transformers | **5/5 = 100%** | 4.0 |
| `resnet18_cifar100` (79.26) | vision | hard — registration helper + timm | **4/5 = 80%** | 13.0 |

All matched runs reproduced the published number **exactly** (abs_diff = 0).
**Difficulty, not luck, is the variable.** The hard oracle first failed 0/2
(rabbit-holing into manual architecture reconstruction); a *transferable*
strategy fix — read the model card prose; "not registered" → install+import the
helper — moved it **0 → 80%**.

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
- **Verification** is deterministic: the LLM extracts the number (with the log
  line as evidence), but the comparison is plain code.
- **Evaluation** reports staged pass rates (`repo_inspected → … → claim_matched`)
  + a failure taxonomy; `eligibility` vs `outcome` are separated so a task can't
  be moved out of the denominator after the agent fails.

## Repo layout

```
agent/       LLM + ReAct loop + tiered self-repair
exec/        persistent session (shell/file actions, replay log)
verify/      deterministic metric extraction + comparison
retrieval/   navigation ladder: corpus · keyword/BM25/dense/hybrid/rerank
evals/       benchmark manifests · significance (clustered bootstrap) · RESULTS.md
run_repro.py · run_reliability.py   # run one manifest / measure reliability
```

## Setup & run

Python 3.12, an OpenAI-compatible chat key (DeepSeek) + DashScope key for
embeddings, in a gitignored `.env`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env:  LLM_API_KEY=…  LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-chat
#        DASHSCOPE_API_KEY=…  DASHSCOPE_BASE_URL=…  EMBEDDING_MODEL=text-embedding-v4

python run_repro.py evals/benchmark/cifar10_resnet20.yaml   # reproduce one oracle
python run_reliability.py evals/benchmark/resnet18_cifar100.yaml 5
python -m retrieval.eval_nav --dense                        # the navigation ladder
pytest tests/
```

## Honest status & caveats

- **Reproducibility crisis, as data:** `RepDistiller` = `artifact_blocked` (dead
  checkpoint host); `mmpretrain` full reproduction = `env_blocked` (mmcv won't
  build on Py3.12) — both reported, neither counted as agent failure.
- Oracles so far are **library-load**; a true clone-and-navigate reproduction is
  blocked by the above. Navigation is measured on the cloned source instead.
- Small n (5–8 runs/oracle), single model — effect sizes + staged rates, not
  significance claims.
- Execution is a subprocess session, **not** the Docker/VM isolation untrusted
  repos need (security debt, deferred). Don't point it at code you don't trust.
