# Benchmark freeze manifest

Provenance for the reproduction-task suite so a run can be repeated and audited.
Pin these before reporting; do not float on `main`/`master`.

## Agent

| field | value |
|---|---|
| agent commit | `1d9a09fbd79916ec54d3e728605ad7491fcb076b` (record the HEAD you ran on) |
| LLM | `deepseek-chat` (OpenAI-compatible), `temperature` per `agent/llm.py` |
| embeddings | DashScope `text-embedding-v4` |
| execution budget | ≤5 evaluations per looped condition (`MAX_REPAIR_ROUNDS = 4`) |

## Benchmark repositories (pinned commit SHAs)

| repo | commit SHA |
|---|---|
| OpenOOD | `3c35632ee91b54b09d1f085d04f94744cece7d0b` |
| mmpretrain | `ee7f2e88501f61aa95c742dd5f429f039935ee90` |
| robustbench | `78fcc9e48a07a861268f295a777b975f25155964` |
| RepDistiller (artifact-blocked) | `b84f547c5db6a35318d4671d7d5c4de74c822403` |

## Checkpoints (SHA-256, first 12 hex)

| artifact | sha256[:12] |
|---|---|
| mmpretrain ResNet-18 b16x8 CIFAR-10 (`repos/mmpretrain_assets/ckpt.pth`) | `bd6371c8f499` (matches the official `…-bd6371c8.pth`) |
| RobustBench `Carmon2019Unlabeled.pt` | `f3ea703e4e98` |
| DistilBERT SST-2, detectors ResNet-18/VGG16 | HuggingFace Hub snapshot (pinned by the cached commit hash under `~/.cache/huggingface`) |

## Tasks and blind level

`strict` = the published target is absent from the agent's workspace; `soft` =
present in the public repo (never surfaced by the task or verifier), the agent
must still run the real eval to produce it.

| task | target | num_examples | blind |
|---|---|---|---|
| DistilBERT SST-2 | 91.06 acc | 872 | strict |
| detectors ResNet-18 CIFAR-100 | 79.26 top-1 | 10000 | strict |
| OpenOOD EBO Near-OOD AUROC | 87.58 | 9000 + 7793 ×3 runs | strict |
| RobustBench Carmon2019 (AutoAttack, n=50) | 52.0 robust acc | 50 | strict (README `52.00%` scrubbed at provisioning) |
| mmpretrain ResNet-18 CIFAR-10 | 94.82 top-1 | 10000 | **soft** (94.82 is in the repo's own model-zoo metafile) |

## How to reproduce a run

```bash
# strict-blind verification: the agent never sees the target; only structured
# evidence from a real eval (provenance bound to the emitting command) can match.
python run_distilbert_multi_rag.py                 # full pipeline
PIPELINE=solo-repair python run_openood_multi_rag.py
python evals/report_tables.py                       # regenerate the result tables
```

## Honest reproducibility caveats (still open)

- The HuggingFace caches are pinned only by whatever commit currently sits in
  `~/.cache/huggingface`; a fully held-out run should pin those snapshot hashes
  explicitly and rebuild the Docker images from a locked dependency file.
- Docker images (`repro-openood`, `repro-mmpretrain`) are built once locally; their
  digests are not yet recorded here.
- N=5 per cell; results are indicative, not significance-tested. All five tasks
  are **development** tasks (prompts were iterated against them) — there is no
  held-out split, so generalization claims are scoped to this suite.
