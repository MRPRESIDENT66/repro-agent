# Generic Reproduction Results

Single generic agent (no task-specific prompts), blind reproduction, verifier-side
recomputation. Each task gives the agent only a public task description, a public
execution command, and a public result-artifact contract. The agent never sees the
target number; a deterministic verifier recomputes the metric from the agent's
per-sample `predictions.json` against gold labels it loads itself.

| | |
|---|---|
| Branch / commit | `generic-only` |
| LLM | `deepseek-chat` |
| Sampling | E1 is an archived N=5 summary; main E2 uses N=5; OpenOOD supplemental uses N=3 |
| Verifier | independent recomputation from `predictions.json` (printed numbers ignored) |

## Main Results

Unless marked supplemental, each cell summarizes five independent LLM runs under
the same model, prompts, execution budget, and verifier. We report pass@k, mean
command count, mean evaluation executions, mean LLM cost, and verifier-level
failure modes.

### E1 — Coverage N=5 

This table is retained from the earlier full-pipeline N=5 coverage run. It is
useful for the project narrative, but should be treated as an archived summary
rather than a freshly regenerated local table.

| Task | pass@5 | mean cmds | mean evals | mean cost | failure modes                                                       |
|---|-------:|---:|---:|---:|---------------------------------------------------------------------|
| DistilBERT SST-2 |    5/5 | 2.00 | 1.00 | ¥0.049 | —                                                                   |
| detectors RN18 / CIFAR-100 |    4/5 | 2.40 | 1.20 | ¥0.068 | outside_tolerance×1                                                 |
| detectors VGG16-bn / CIFAR-10 |    5/5 | 2.80 | 1.40 | ¥0.083 | —                                                                   |
| mmpretrain RN18 / CIFAR-10 |    3/5 | 7.20 | 3.60 | ¥0.409 | no_recomputable_predictions×1, outside_tolerance×1 |
| OpenOOD EBO AUROC |    2/5 | 5.60 | 2.80 | ¥0.432 | workflow_error×2, no_recomputable_predictions×1                     |
| RobustBench Carmon2019 |    4/5 | 5.60 | 2.80 | ¥0.370 | outside_tolerance×1                                                 |

### E2 — Pipeline Ablation N=5

The current ablation uses three budget-fair conditions: `solo`, `solo-repair`,
and `full`. DistilBERT and detectors RN18 were run five times per condition.

| Task | Condition | pass@5 | mean cmds | mean evals | mean cost | failure modes |
|---|---|---:|---:|---:|---:|---|
| DistilBERT SST-2 | `solo` | 5/5 | 2.00 | 1.00 | ¥0.008 | — |
| DistilBERT SST-2 | `solo-repair` | 5/5 | 2.00 | 1.00 | ¥0.008 | — |
| DistilBERT SST-2 | `full` | 5/5 | 2.00 | 1.00 | ¥0.050 | — |
| detectors RN18 / CIFAR-100 | `solo` | 0/5 | 2.00 | 1.00 | ¥0.010 | no_recomputable_predictions×5 |
| detectors RN18 / CIFAR-100 | `solo-repair` | 5/5 | 4.80 | 2.40 | ¥0.045 | — |
| detectors RN18 / CIFAR-100 | `full` | 5/5 | 2.40 | 1.20 | ¥0.079 | — |

### OpenOOD EBO Supplemental Stress Test N=3

OpenOOD EBO was added as a harder supplemental stress test after restoring the
external repo/data/checkpoint assets. This is **not** pass@5 yet: three runs per
condition are complete, with two remaining if we want to match the main E2 budget.

| Task | Condition | pass@3 | mean cmds | mean evals | mean cost | failure modes |
|---|---|---:|---:|---:|---:|---|
| OpenOOD EBO AUROC | `solo` | 0/3 | 2.00 | 1.00 | ¥0.040 | no_recomputable_predictions×3 |
| OpenOOD EBO AUROC | `solo-repair` | 0/3 | 9.33 | 4.67 | ¥0.323 | outside_tolerance×3 |
| OpenOOD EBO AUROC | `full` | 1/3 | 9.33 | 4.67 | ¥0.601 | outside_tolerance×1, no_recomputable_predictions×1 |

The OpenOOD runs show a sharper separation than the easier tasks: one-shot
generation consistently fails before producing a recomputable artifact, repair
can produce correctly shaped artifacts but misses the AUROC semantics, and full
collaboration has one exact verifier pass so far but is not stable yet at N=3.

## Notes

- Costs are LLM token costs in RMB; reproduction inference runs on local CPU/Docker.
- The main E2 ablation intentionally keeps only three conditions: `solo`,
  `solo-repair`, and `full`.
- OpenOOD is reported separately because it is a harder supplemental stress case
  and is not yet budget-matched to the main N=5 ablation.
