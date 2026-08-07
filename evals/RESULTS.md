# Repro-Agent Evaluation Results

Fresh blind-reproduction experiments run on 2026-08-07. Every cell contains five
independent DeepSeek V4 Flash runs with thinking disabled, temperature 0, fixed
generic prompts, one initial execution plus at most four repairs, and the same
task verifier. The agent never receives the hidden target or verifier gold.

Two complementary measures are reported:

- **verified**: the independent verifier accepted a metric recomputed from the
  run's complete per-sample artifact.
- **no workflow error / verified**: among verifier-accepted runs, the fraction
  that also completed without an orchestration exception. This conditional rate
  exposes role/handoff reliability separately from task correctness.

Costs are estimates from configured token rates, not provider billing records.
`mean evals` is audited from each run's replayable `commands.sh`; this recovers
executions followed by a Reviewer synthesis exception.

## Aggregate Finding

Across the four-task ablation (20 runs per condition), execution-grounded repair
doubled verified success from **7/20 (35%)** for one-shot `solo` to **14/20
(70%)** for `solo-repair`. The `full` multi-role pipeline reached **17/20 (85%)**
verified success. Among those 17 accepted runs, **12/17 (70.6%)** had no workflow
error; Reviewer synthesis/handoff failures sometimes occurred after a valid
artifact had already been produced.

This is the central result: repair feedback clearly helps; role specialization
raises the verified ceiling on the hardest tasks, but extra coordination also
introduces a measurable reliability and cost tax.

### Adaptive Mode Implementation Smoke (N=1)

The new `adaptive` condition is not included in the aggregate claims above until
it has the same N=5 coverage. A DistilBERT smoke run verified **91.055% over all
872 examples** with no workflow error. Router skipped Navigator and Critic; two
repeated `missing_artifact` failures triggered Auditor only after the repetition,
and the third evaluation passed. The run used 3 evaluations and cost ¥0.0666.
This validates conditional routing and final private verification, but is not
enough evidence to claim lower average cost or higher accuracy than `full`.

<!-- GENERATED_N5_START -->
## E1: Six-Task Coverage N=5

For the four ablation tasks this table uses the `full` rows from E2; detectors VGG16 and RobustBench are additional full-pipeline N=5 coverage cells. All 30 runs use the same frozen agent snapshot and LLM configuration.

| Task | verified | no workflow error / verified | mean cmds | mean evals | mean cost | failure modes |
|---|---:|---:|---:|---:|---:|---|
| DistilBERT SST-2 | 5/5 | 4/5 (80%) | 2.00 | 1.00 | ¥0.062 | workflow_error×1 |
| detectors RN18 / CIFAR-100 | 3/5 | 1/3 (33%) | 2.40 | 1.20 | ¥0.076 | workflow_error×4 |
| detectors VGG16-bn / CIFAR-10 | 5/5 | 4/5 (80%) | 3.20 | 1.60 | ¥0.113 | workflow_error×1 |
| mmpretrain RN18 / CIFAR-10 | 5/5 | 4/5 (80%) | 4.80 | 2.40 | ¥0.297 | workflow_error×1 |
| OpenOOD EBO AUROC | 4/5 | 3/4 (75%) | 7.60 | 3.80 | ¥0.544 | workflow_error×1, outside_tolerance×1 |
| RobustBench Carmon2019 | 5/5 | 5/5 (100%) | 7.20 | 3.60 | ¥0.484 | — |
| **Total** | **27/30** | **21/27 (78%)** | — | — | **¥7.881 total** | — |

## E2: Pipeline Ablation N=5

| Task | Condition | verified | no workflow error / verified | mean cmds | mean evals | mean cost | failure modes |
|---|---|---:|---:|---:|---:|---:|---|
| DistilBERT SST-2 | `solo` | 5/5 | 5/5 (100%) | 2.00 | 1.00 | ¥0.011 | — |
| DistilBERT SST-2 | `solo-repair` | 5/5 | 5/5 (100%) | 2.00 | 1.00 | ¥0.009 | — |
| DistilBERT SST-2 | `full` | 5/5 | 4/5 (80%) | 2.00 | 1.00 | ¥0.062 | workflow_error×1 |
| detectors RN18 / CIFAR-100 | `solo` | 1/5 | 1/1 (100%) | 2.00 | 1.00 | ¥0.018 | no_recomputable_predictions×4 |
| detectors RN18 / CIFAR-100 | `solo-repair` | 3/5 | 3/3 (100%) | 4.00 | 2.00 | ¥0.039 | outside_tolerance×2 |
| detectors RN18 / CIFAR-100 | `full` | 3/5 | 1/3 (33%) | 2.40 | 1.20 | ¥0.076 | workflow_error×4 |
| mmpretrain RN18 / CIFAR-10 | `solo` | 1/5 | 1/1 (100%) | 1.60 | 0.80 | ¥0.024 | no_recomputable_predictions×2, outside_tolerance×1, workflow_error×1 |
| mmpretrain RN18 / CIFAR-10 | `solo-repair` | 4/5 | 4/4 (100%) | 3.60 | 1.80 | ¥0.078 | workflow_error×1 |
| mmpretrain RN18 / CIFAR-10 | `full` | 5/5 | 4/5 (80%) | 4.80 | 2.40 | ¥0.297 | workflow_error×1 |
| OpenOOD EBO AUROC | `solo` | 0/5 | — | 1.80 | 1.00 | ¥0.022 | no_recomputable_predictions×5 |
| OpenOOD EBO AUROC | `solo-repair` | 2/5 | 2/2 (100%) | 6.40 | 3.20 | ¥0.177 | outside_tolerance×3 |
| OpenOOD EBO AUROC | `full` | 4/5 | 3/4 (75%) | 7.60 | 3.80 | ¥0.544 | workflow_error×1, outside_tolerance×1 |

| Condition | verified | no workflow error / verified | mean cmds | mean evals | mean cost |
|---|---:|---:|---:|---:|---:|
| `solo` | **7/20 (35%)** | **7/7 (100%)** | 1.85 | 0.95 | ¥0.019 |
| `solo-repair` | **14/20 (70%)** | **14/14 (100%)** | 4.00 | 2.00 | ¥0.076 |
| `full` | **17/20 (85%)** | **12/17 (71%)** | 4.20 | 2.10 | ¥0.245 |
<!-- GENERATED_N5_END -->

## E3: Post-Freeze Held-Out Repository N=5

After the development-task runtime and prompts were committed as `8fa152e`, a
new repository and metric were selected: Sentence-Transformers v5.7.0 with the
pinned `all-mpnet-base-v2` checkpoint on the 1,379-pair STS-B test split. Only
evaluation-side files were added: public task/workspace provisioning, private
gold and Spearman recomputation, an asset-preparation script, a thin runner, and
Oracle tests. The frozen `agent/`, `retrieval/`, `exec/`, and `verify/` paths were
unchanged.

The agent received sentence pairs without labels and wrote one cosine similarity
per pair. The verifier independently recomputed Spearman correlation against
private gold scores. All five runs produced the full artifact in one evaluation;
no Repair round was needed.

| Task | Condition | verified | no workflow error / verified | mean cmds | mean evals | mean cost |
|---|---|---:|---:|---:|---:|---:|
| Sentence-Transformers `all-mpnet-base-v2` / STS-B test | `full` | **5/5 (100%)** | **5/5 (100%)** | 2.00 | 1.00 | ¥0.200 |

The pinned CPU reference was `0.8342216679`; every reportable run recomputed
`0.8342216139`, well within the predeclared `0.001` tolerance. Configured token
cost was ¥1.0015 total across the five runs.

### Excluded Pilot Disclosure

Manifest `logs/holdout_n5/manifest_20260807T103345Z.json` is excluded from E3.
The first adapter wording exposed `predictions.json` as a literal generic code
marker, so a valid program that honored the public `--output predictions.json`
CLI via `args.output` was falsely rejected before execution. That was an Oracle
adapter defect, not a model-evaluation failure. The protocol wording was fixed,
a regression test was added, and all five attempts were rerun under new names in
`manifest_20260807T104352Z.json`; no agent/runtime/prompt code changed. The pilot
artifacts remain locally auditable and were not selected into the reported cell.

## Failure Analysis

- **One-shot misses the hard tasks.** OpenOOD `solo` produced no recomputable
  artifact in 5/5 runs; detectors and mmpretrain showed the same failure class.
- **Repair fixes execution, not every semantic error.** `solo-repair` doubled
  aggregate success, but wrong preprocessing, label mapping, or score direction
  could still produce complete artifacts outside tolerance.
- **Full improves verified outcomes but is coordination-sensitive.** Its 17/20
  verifier result is best, while only 12/17 verified runs had no workflow error.
  The other five were mostly Reviewer report validation/synthesis failures after
  valid model inference.
- **Easy tasks saturate.** All three DistilBERT conditions passed 5/5, so added
  roles only increased cost and failure surface there.

## OpenOOD Case Study

OpenOOD is the clearest repair example. Early scripts used adjacent training
normalization instead of the Evaluation API's test preprocessing, omitted
Resize/CenterCrop, or emitted EBO scores with inverted polarity. Source-grounded
Reviewer/Repair rounds traced the canonical preprocessor and changed the score to
the required higher-is-OOD convention. The final verifier recomputed AUROC from
50,379 per-sample OOD scores per run; `full` improved from `solo` **0/5** to
**4/5** verified.

OpenOOD used the host Apple MPS backend for speed. A complete MPS/CPU parity run
produced 87.5822776 versus 87.5822775. It used the same complete sample contract
and verifier, but host MPS has weaker execution isolation than offline Docker.

## Reproducibility

- Agent base commit: `65dde088dcafda1f80325df1c6a885f51f2d9b80`
- Frozen experiment diff SHA-256:
  `c3e9bc0e014505236c4dac466c85bbaf8bfd614af3806720edbd23cea9d9c7b1`
- Main manifest: `logs/n5/manifest_20260807T015201Z.json`
- Coverage manifests: `logs/n5/manifest_20260807T052105Z.json`,
  `manifest_20260807T060206Z.json`, `manifest_20260807T061133Z.json`
- Held-out manifest: `logs/holdout_n5/manifest_20260807T104352Z.json`
- Held-out frozen agent commit: `8fa152e`; Sentence-Transformers source:
  `b2a9529cf6312d2b2a8ffa2b64d82fabc1571bd8` (`v5.7.0`)
- STS-B dataset revision: `ab7a5ac0e35aa22088bdcf23e7fd99b220e53308`;
  all-mpnet-base-v2 revision: `e8c3b32edf5434bc2275fc9bab85f82640a19130`
- OpenOOD backend: host MPS; mmpretrain: offline Docker/CPU; remaining tasks:
  clean local subprocess environments with pre-cached assets.

The worktree was not clean, so the runner saved the exact binary diff beside each
manifest. Results should be attributed to base commit + diff hash, not to the base
commit alone.

## Limits

- N=5 is prototype evidence, not a significance-tested benchmark.
- Six tasks were used while iterating the runtime. E3 adds one post-freeze
  held-out repository, which is useful evidence but not a representative set.
- Every new task still requires a hand-written public adapter and hidden verifier.
- Full collaboration is a configurable mode, not a claim of universal dominance.
