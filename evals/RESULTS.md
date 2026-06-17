# Generic Reproduction Results

Single generic agent (no task-specific prompts), blind reproduction, verifier-side
recomputation. Each task gives the agent only a public task description, a public
execution command, and a public result-artifact contract. The agent never sees the
target number; a deterministic verifier recomputes the metric from the agent's
per-sample `predictions.json` against gold labels it loads itself.

| | |
|---|---|
| Branch / commit | `generic-only` @ `9777587` |
| LLM | `deepseek-chat` |
| Sampling | N=1 probe per cell |
| Verifier | independent recomputation from `predictions.json` (printed numbers ignored) |

## E1 — Coverage (full pipeline)

All six task instances, full orchestration (Navigator → Reproducer → Critic →
execute → Reviewer → repair loop), shared budget of ≤5 executions.

| Task | Backend | Match | Recomputed | Target | abs_diff | Cmds | Cost |
|---|---|:--:|---|---|---|:--:|---|
| DistilBERT SST-2 | subprocess | ✅ | 91.055 | 91.06 | 0.005 | 2 | ¥0.10 |
| detectors ResNet18 / CIFAR-100 | subprocess | ✅ | 79.26 | 79.26 | 0.00 | 2 | ¥0.07 |
| detectors VGG16-bn / CIFAR-10 | subprocess | ✅ | 93.37 | 93.37 | 0.00 | 4 | ¥0.13 |
| mmpretrain ResNet18 / CIFAR-10 | Docker amd64 | ✅ | 94.82 | 94.82 | 0.00 | 2 | ¥0.14 |
| OpenOOD EBO (Near-OOD AUROC) | subprocess | ✅ | 87.582 | 87.58 | 0.002 | 4 | ¥0.29 |
| RobustBench Carmon2019 (robust acc) | subprocess | ❌ | 58.0 | 52.0 | 6.0 | 4 | ¥0.29 |

**5 / 6 reproduced blind**, all exact or near-exact, each recomputed by the verifier
from per-sample predictions. The two cross-backend points (Docker amd64 mmpretrain,
the OpenOOD multi-dataset AUROC) both pass under the same generic agent.

### Failure mode is fail-closed
The single failure is `outside_tolerance` — the verifier ran, recomputed, and
rejected a wrong number. No task produced a forged or aggregate-only pass.

## E2 — Pipeline ablation

Two representative tasks — DistilBERT (easy) and detectors ResNet18/CIFAR-100 (a
task with an `import detectors` timm-registration gotcha) — across five
budget-fair conditions (same ≤5-execution budget, differing only in orchestration
depth and post-execution feedback).

| Condition | DistilBERT | detectors RN18 | Failure mode (when ❌) |
|---|:--:|:--:|---|
| `solo` (one shot, no feedback) | ❌ | ❌ | `no_recomputable_predictions` |
| `solo-retry` (blind re-generate) | ✅ | ❌ (burned 10 cmds) | retry cannot clear the gotcha |
| `solo-repair` (single agent + error feedback) | ✅ | ✅ | — |
| `team` (Navigator + Critic, no repair) | ✅ | ✅ | — |
| `full` | ✅ | ✅ | — |

### Reading
- **`solo` fails on both** — a single generation does not even emit a valid
  `predictions.json` artifact.
- **Blind retry is not enough** — it recovers the easy task but cannot overcome
  the registration gotcha; the agent re-generates similar broken code and exhausts
  the budget (10 commands).
- **What works is feedback or planning** — `solo-repair` (learn from the real
  execution error) and `team` (better up-front plan) both clear the gotcha.

The take-away: on a non-trivial task, reproduction succeeds not by trying more
times but by reading the real execution error.

## RobustBench failure — root cause

The RobustBench point now **completes** (the AutoAttack CPU budget was raised to
2700s; it no longer times out) but lands at robust accuracy 58.0 vs target 52.0 —
29/50 vs 26/50 robust, a 3-sample overshoot in the *weaker-attack* direction.

The agent got the hard parts right: the custom AutoAttack ensemble
(`apgd-ce` + `apgd-dlr`, 1 restart), the sample count (first 50), and the epsilon
(8/255). It got one convention wrong: it applied `transforms.Normalize` to the
**inputs** before AutoAttack, with a code comment asserting the model has no
internal normalization. RobustBench's protocol is the opposite — `load_clean_dataset`
returns images in `[0,1]` and `load_model` returns a normalization-wrapped model.
By moving normalization outside the model, the agent measures the L∞ ε-ball in
*normalized* space (divided by σ≈0.25), so the effective pixel-space perturbation
is ~4× smaller — a weaker attack, hence robust accuracy too high.

The public contract is not at fault: it states the first 50 examples, the epsilon,
and the exact attack ensemble. The input-space convention is something the agent
must read from the repository (`load_clean_dataset` / `load_model`); spelling it out
in the contract would amount to feeding the answer to this specific bug, which the
generic setting deliberately avoids.

## Notes

- Costs are LLM token costs in RMB; reproduction inference runs on local CPU/Docker.
- The DistilBERT and detectors-RN18 `full` rows are reused as the `full` condition
  in E2.
- This is N=1 probe coverage; pass/fail is determined by the verifier, but pass
  *rates* would need repeated runs to estimate.

## Main N=5 Results

Each cell is five independent LLM runs under the same model, prompts, execution budget, and verifier. We report pass@5, mean command count, mean evaluation executions, mean LLM cost, and verifier-level failure modes.

### E1 — Coverage N=5

| Task | pass@5 | mean cmds | mean evals | mean cost | failure modes |
|---|---:|---:|---:|---:|---|
| DistilBERT SST-2 | 5/5 | 2.00 | 1.00 | ¥0.049 | — |
| detectors RN18 / CIFAR-100 | 4/5 | 2.40 | 1.20 | ¥0.068 | outside_tolerance×1 |
| detectors VGG16-bn / CIFAR-10 | 5/5 | 2.80 | 1.40 | ¥0.083 | — |
| mmpretrain RN18 / CIFAR-10 | 2/5 | 7.20 | 3.60 | ¥0.409 | no_recomputable_predictions×1, workflow_error×1, outside_tolerance×1 |
| OpenOOD EBO AUROC | 0/5 | 5.60 | 2.80 | ¥0.432 | workflow_error×4, no_recomputable_predictions×1 |
| RobustBench Carmon2019 | 4/5 | 5.60 | 2.80 | ¥0.370 | outside_tolerance×1 |

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

### E2 — OpenOOD EBO Supplemental Ablation N=3

OpenOOD EBO was added as a harder supplemental ablation after restoring the
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

### E1 — Coverage N=2 Partial Rerun

Two fresh full-pipeline runs have been completed for each E1 task. This is a
partial rerun toward N=5; three more runs remain.

| Task | pass@2 | mean cmds | mean evals | mean cost | failure modes |
|---|---:|---:|---:|---:|---|
| DistilBERT SST-2 | 2/2 | 2.00 | 1.00 | ¥0.047 | — |
| detectors RN18 / CIFAR-100 | 1/2 | 4.00 | 2.00 | ¥0.106 | outside_tolerance×1 |
| detectors VGG16-bn / CIFAR-10 | 2/2 | 2.00 | 1.00 | ¥0.066 | — |
| mmpretrain RN18 / CIFAR-10 | 2/2 | 8.00 | 4.00 | ¥0.535 | — |
| OpenOOD EBO AUROC | 0/2 | 9.00 | 4.50 | ¥0.558 | outside_tolerance×2 |
| RobustBench Carmon2019 | 2/2 | 4.00 | 2.00 | ¥0.282 | — |

Note: the earlier successful OpenOOD `full` run
(`e_openood_full_after_restore`, verifier recomputed 87.5823 vs target 87.58)
is counted in the OpenOOD E2 supplemental ablation above. It is not included in
this E1 partial-rerun table because the E1 rows here only count the two fresh
`e1_n5_s{1,2}_full` runs. If we pool all completed OpenOOD `full` runs regardless
of bookkeeping label, OpenOOD is 1/5 overall: one pass
(`e_openood_full_after_restore`) and four failures (`e_openood_full_s2`,
`e_openood_full_s3`, `e1_n5_s1_full`, `e1_n5_s2_full`).
