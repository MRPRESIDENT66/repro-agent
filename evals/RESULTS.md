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

## Honest caveats

- Small n (5–8 runs/oracle); single model (deepseek-chat). Numbers are
  effect-size + pass-rate, not significance claims.
- These oracles are **library-load** artifacts; a true *clone-and-navigate*
  large-repo oracle (where RAG would earn its place) is not yet in the set —
  the best candidate (RepDistiller) was artifact-blocked.
- Execution is a subprocess session (MPS), not the Docker/VM isolation the
  design specifies for untrusted repos (security debt, deferred).
