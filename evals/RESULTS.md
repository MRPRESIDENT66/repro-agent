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

## Multi-agent vs single (M5, isolation ablation)

Reproduce 3 CIFAR-10 ResNets (resnet20/32/56) from one repo. Multi-agent: Lead
splits into 3, each Reproducer runs in an **isolated** context, a deterministic
Verifier checks each. Single: one agent does all 3 in one shared context.

| mode | agents | matched | max context (msgs / chars) | steps |
|---|---|---|---|---|
| multi | 3 | 3/3 | **26 / 21.5k** | 18 |
| single | 1 | 3/3 | 41 / 26.9k | 20 |

**Finding:** identical success (3/3 both) — multi-agent does **not** improve
success or speed on independent easy sub-tasks. Its only measurable benefit is
the designed **isolation**: each agent's context stays ~30–55% smaller than the
single agent juggling all three. That isolation earns its place only when
sub-tasks are long/noisy enough that a shared context would degrade — these
aren't. Reported as-is (the design said: value is isolation, not success;
report n.s. if so).

## Context compression (M4)

The debug trajectory grows over many repair cycles. Compression keeps
system+task+the last few turns full and shrinks older observations/tracebacks.

- On a representative 15-step env-hell trajectory: **27.3k → 11.1k chars (−59%)**,
  recent context untouched (unit-tested).
- Live ablation on `resnet18_cifar100` was **inconclusive**: both compress on/off
  reproduced (success unaffected), but the agent happened to solve it in 4–5
  steps — a *short* trajectory, so compression never engaged. Reported as-is:
  compression bounds context on **long** debug sequences (env hell); on quick
  reproductions it's a no-op.

## Repo-search tool wired into the loop

The agent can emit a ` ```search ` block (a natural-language query) once it has
cloned a large repo; it returns the most relevant files via **BM25 + LLM rerank
(no embeddings — the ladder showed dense doesn't help)**. Verified standalone on
mmpretrain ("evaluate resnet18 on cifar10" → the right config ranks #1). Only
exercised when an oracle requires cloning a large repo (currently env-blocked).

## Honest caveats

- Small n (5–8 runs/oracle); single model (deepseek-chat). Numbers are
  effect-size + pass-rate, not significance claims.
- These oracles are **library-load** artifacts; a true *clone-and-navigate*
  large-repo oracle (where RAG would earn its place) is not yet in the set —
  the best candidate (RepDistiller) was artifact-blocked.
- Execution is a subprocess session (MPS), not the Docker/VM isolation the
  design specifies for untrusted repos (security debt, deferred).
