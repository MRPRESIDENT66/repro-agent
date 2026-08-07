# Benchmark Freeze Manifest

This file records the fixed assets and runtime assumptions behind reported
results. Every newly reported experiment should also record the exact clean
agent commit, run date, model endpoint/version, and Docker image digest.

## Agent Runtime

| Field | Value |
|---|---|
| LLM | `deepseek-v4-flash` through an OpenAI-compatible endpoint |
| Thinking | disabled |
| Sampling | `temperature=0` (provider-side nondeterminism may remain) |
| Execution budget | One initial execution plus at most four repair executions |
| Execution backend | Recorded per run; OpenOOD supports `docker` (CPU, offline) and `mps` (host Apple GPU, weaker isolation) |
| Current local base commit | `65dde088dcafda1f80325df1c6a885f51f2d9b80` |
| Formal-run diff SHA-256 | `c3e9bc0e014505236c4dac466c85bbaf8bfd614af3806720edbd23cea9d9c7b1` |
| Main formal manifest | `logs/n5/manifest_20260807T015201Z.json` |
| Coverage manifests | `20260807T052105Z`, `20260807T060206Z`, `20260807T061133Z` |

The formal worktree was not clean. The runner saved its exact binary patch beside
the manifests; attribute results to the base commit plus diff hash, not to the
base commit by itself.

## Benchmark Repositories

| Repository | Commit SHA |
|---|---|
| OpenOOD | `3c35632ee91b54b09d1f085d04f94744cece7d0b` |
| mmpretrain | `ee7f2e88501f61aa95c742dd5f429f039935ee90` |
| robustbench | `78fcc9e48a07a861268f295a777b975f25155964` |
| RepDistiller (artifact-blocked) | `b84f547c5db6a35318d4671d7d5c4de74c822403` |

## Checkpoints

| Artifact | Pin |
|---|---|
| mmpretrain ResNet-18 CIFAR-10 | official `resnet18_b16x8_cifar10_20210528-bd6371c8.pth`; SHA-256 prefix `bd6371c8f499` |
| RobustBench `Carmon2019Unlabeled.pt` | SHA-256 `f3ea703e4e98d26947bced9580f63922e31423233bbe45eebff8c7d45b7eacfc` |
| DistilBERT and detectors models | HuggingFace snapshot cache; record the exact snapshot hash for future published runs |

## Local Docker Images Used For The Next Run

| Image | Local immutable image ID |
|---|---|
| `repro-openood:latest` | `sha256:5e6d9ab9b193a76fce6f1840bce6b2f83f004813b459f50e71f417a9ab4176e5` |
| `repro-mmpretrain:latest` | `sha256:6debf65941f350983af2346678120f2ab36209b0130b2d1317a794c005d3f262` |

## Honest Limits

- Rebuilding either Docker image requires recording its new image ID; tags alone
  are not immutable.
- OpenOOD MPS and Docker use the same complete sample contract and verifier, but
  MPS executes generated code on the host and must not be described as container
  isolation. Do not mix unlabelled backend results in one comparison table.
- The historical N=5 tables are prototype evidence, not significance-tested
  benchmark conclusions.
- These are development tasks: prompts and runtime behavior were iterated while
  investigating them, so they are not a held-out generalization set.
