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

The later `legacy-pipelines-v1` tag points to the last committed source tree that
still exposes `solo`, `solo-repair`, `full`, and `adaptive`. It is a convenient
code archive, not a replacement for the formal base-commit + diff-hash identity
above.

## Post-Freeze Held-Out Task

| Field | Value |
|---|---|
| Frozen generic agent commit | `8fa152e` |
| Frozen paths | `agent/`, `retrieval/`, `exec/`, `verify/` |
| Task | Sentence-Transformers `all-mpnet-base-v2` on STS-B test |
| Pipeline / repeats | `full`, N=5 |
| Formal manifest | `logs/holdout_n5/manifest_20260807T104352Z.json` |
| Result | 5/5 verifier passes; 5/5 without workflow error |
| Excluded pilot | `manifest_20260807T103345Z.json`; Oracle CLI-output marker defect |

The excluded pilot remains auditable but is not part of the reported E3 cell.
The replacement batch uses distinct attempt names and records the exclusion
reason and hashes of every evaluation-side adapter file.

## Second Held-Out Task: Manifest-Only Adaptive CLIP

| Field | Value |
|---|---|
| Frozen generic runtime | `1f4cdda` (`heldout-v2-freeze`) |
| Frozen task declaration | `e750069` (`heldout-v2-task-freeze`) |
| Frozen runtime paths | `agent/`, `retrieval/`, `exec/`, `verify/`, reusable manifest framework |
| Task | OpenAI CLIP ViT-B/32 zero-shot on complete CIFAR-10 test split |
| Integration | one YAML manifest; no task hook |
| Pipeline / repeats | `adaptive`, N=5 |
| Attempt IDs | `heldout2_n5_s1` through `heldout2_n5_s5` |
| Result | 5/5 verifier passes; 2/5 without workflow error |
| Excluded pilots | `heldout2_pilot_n1`, underspecified ensemble; `heldout2_spec_pilot_n1`, pre-freeze specification check |

The formal batch produced four 89.87% results and one 89.88% result from 10,000
predictions per run. One workflow recovered from a missing artifact through one
Repair execution. Three other predictions passed private verification but are
not workflow-clean because Auditor failed strict structured-report synthesis.
An initial launcher import failure created no attempt and made no LLM or
evaluation call; the unchanged frozen batch was relaunched with `PYTHONPATH=.`.

## Benchmark Repositories

| Repository | Commit SHA |
|---|---|
| OpenOOD | `3c35632ee91b54b09d1f085d04f94744cece7d0b` |
| mmpretrain | `ee7f2e88501f61aa95c742dd5f429f039935ee90` |
| robustbench | `78fcc9e48a07a861268f295a777b975f25155964` |
| sentence-transformers v5.7.0 (held out) | `b2a9529cf6312d2b2a8ffa2b64d82fabc1571bd8` |
| OpenAI CLIP (second held out) | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| RepDistiller (artifact-blocked) | `b84f547c5db6a35318d4671d7d5c4de74c822403` |

## Checkpoints

| Artifact | Pin |
|---|---|
| mmpretrain ResNet-18 CIFAR-10 | official `resnet18_b16x8_cifar10_20210528-bd6371c8.pth`; SHA-256 prefix `bd6371c8f499` |
| RobustBench `Carmon2019Unlabeled.pt` | SHA-256 `f3ea703e4e98d26947bced9580f63922e31423233bbe45eebff8c7d45b7eacfc` |
| Sentence-Transformers `all-mpnet-base-v2` | Hugging Face revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| OpenAI CLIP `ViT-B-32.pt` | SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
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
