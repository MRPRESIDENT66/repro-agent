REVIEW_STATUS: REPAIR_REQUIRED

**Finding: The aggregation logic in `eval_ebo.py` does not match the public-contract specification.**

The handoff states: "The `eval_ood.py` script (lines 171–195) automatically: … Aggregates results across seeds (mean ± std) … Reports per-dataset AUROC (CIFAR-100 and TinyImageNet separately)." The expected output format is a table with per-dataset AUROC (mean ± std across seeds), e.g., `cifar100: XX.XX ± Y.YY` and `tin: XX.XX ± Y.YY`.

However, the custom `eval_ebo.py` implementation computes a single aggregated number: it takes the mean of the two dataset AUROCs within each seed, then averages those means across seeds. This produces `actual = 87.58`, which is a single scalar, not the required per-dataset metrics with standard deviation.

The repository evidence confirms the correct behavior. In `scripts/eval_ood.py` (lines 171–195), the `Evaluator` class (from `openood.evaluation_api`) computes per-dataset metrics (AUROC, etc.) for each seed, then aggregates them across seeds to produce per-dataset mean ± std. The handoff explicitly says "Reports per-dataset AUROC (CIFAR-100 and TinyImageNet separately)."

The current implementation's aggregation (`dataset_mean_then_run_mean`) is a semantic mismatch. The contract requires reporting the mean AUROC for each Near-OOD dataset separately (with std), not a single averaged number.

**Required repair:** Modify the aggregation in `eval_ebo.py` to compute per-dataset mean and std across seeds, and output the results as specified (e.g., `cifar100: XX.XX ± Y.YY`, `tin: XX.XX ± Y.YY`). The `actual` field in the JSON output should be removed or replaced with the per-dataset metrics.

REVIEW_STATUS: REPAIR_REQUIRED
