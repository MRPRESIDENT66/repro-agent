# Benchmark results

Agent: deepseek-chat (OpenAI-compatible) · execution: persistent subprocess
session (MPS) · every match is **anti-hardcode gated** (the eval script must
load the dataset and predict over it, not print a literal) and spot-verified by
re-running the agent's own script independently.

## Reproduction reliability (hint-light tasks)

The agent is told only *what* to reproduce (model + dataset + that a published
value exists) — never *how* (loading mechanism, dataset id, normalization).

| Oracle | Domain | Difficulty | matched | avg steps | avg errors |
|---|---|---|---|---|---|
| `cifar10_resnet20` (92.60) | vision | easy — torch.hub | **8/8 = 100%** | 2.1 | 0.0 |
| `distilbert_sst2` (91.06) | NLP | medium — transformers | **5/5 = 100%** | 4.0 | 0.0 |
| `resnet18_cifar100` (79.26) | vision | hard — `import detectors` + timm | **4/5 = 80%** | 13.0 | 5.2 |

All matched runs reproduced the published value **exactly** (abs_diff = 0.0).

## Difficulty is the variable, not luck

- Easy/medium artifacts (standard loaders) are reproduced **reliably in 2–4
  steps with no errors**.
- The hard artifact (a non-obvious registration-via-helper-library load) costs
  **~13 steps, ~5 errors, and fails 1/5** — the agent must read the model card,
  discover it needs `import detectors`, and avoid rabbit-holing into manual
  architecture reconstruction. A targeted, *transferable* strategy fix (read the
  README prose; "not registered" → install+import the helper) moved this oracle
  from **0 → 80%**.

## Failure taxonomy (observed)

`nonobvious_loading` (registration helper not discovered) · `dataset_id` (HF
ids need `namespace/config`) · `dataset_link_dead` (canonical CIFAR URL 403s) ·
`preprocessing` (wrong normalization → close-but-wrong number) · `label_field`
(CIFAR-100 fine vs coarse).

## Artifact-blocked (excluded from the denominator)

`RepDistiller` (CRD, a stronger *navigation* candidate) — checkpoint host
`shape2prog.csail.mit.edu` is **dead** → `outcome=artifact_blocked`, reported
separately, not counted as an agent failure.

## Repo navigation — retrieval ladder (M3)

Can retrieval locate the eval entry + config in a **large** repo
(`mmpretrain`, 1858 files)? 5 hint-light queries, gold = the file(s) you must
land on; metric = recall@k. (mmpretrain's full reproduction is `env_blocked`
— `mmcv` won't build on Python 3.12 — but navigation is measured on the cloned
source regardless.)

| Rung | recall@5 | recall@10 |
|---|---|---|
| keyword (fair grep: path-weighted presence) | 60% | 60% |
| BM25 | 60% | 60% |
| dense (text-embedding-v4) | 50% | 60% |
| hybrid (BM25 + dense, RRF) | 60% | 60% |
| **+ LLM rerank** | **80%** | **80%** |

**Finding:** for code navigation the *retrieval algorithm barely matters* —
keyword ≈ BM25 ≈ dense ≈ hybrid (~60%), because query terms (model/dataset
names) are literal in paths, and they all miss `tools/test.py` (the entry,
drowned by `slurm_test.sh` / `mim_dist_test.sh` / test docs). **Dense embeddings
do not beat BM25 here.** The win is the **LLM reranker (+20pp)**, which
disambiguates the true entry script from look-alikes — retrieval recalls
candidates, the LLM judges which is the entry. (n=5, illustrative not
significant.)

## Honest caveats

- Small n (5–8 runs/oracle); single model (deepseek-chat). Numbers are
  effect-size + pass-rate, not significance claims.
- These oracles are **library-load** artifacts; a true *clone-and-navigate*
  large-repo oracle (where RAG would earn its place) is not yet in the set —
  the best candidate (RepDistiller) was artifact-blocked.
- Execution is a subprocess session (MPS), not the Docker/VM isolation the
  design specifies for untrusted repos (security debt, deferred).
