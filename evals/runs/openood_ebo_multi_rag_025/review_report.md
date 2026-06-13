REVIEW_STATUS: REPAIR_REQUIRED

**Finding:** The implementation in `eval_ebo.py` uses the wrong sign convention for the EBO energy score when computing AUROC. The official OpenOOD `EBOPostprocessor.postprocess()` returns `conf = temperature * torch.logsumexp(output / temperature, dim=1)` as the confidence score, where **higher confidence means more ID-like**. The `compute_auroc` function in `eval_ebo.py` negates the scores (`-id_scores, -ood_scores`) before computing AUROC, which inverts the ranking and produces incorrect (near-random) AUROC values when the sign is accidentally flipped in some runs.

**Evidence from repository:**

1. **Official EBO postprocessor** (`openood/postprocessors/ebo_postprocessor.py`, lines 24-27):
   ```python
   conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
   return pred, conf
   ```
   The returned `conf` is the energy score directly (higher = more ID-like). This is the standard EBO convention used by the OpenOOD evaluation API.

2. **Official evaluation script** (`scripts/eval_ood.py`) uses the `Evaluator` class which calls `postprocessor.postprocess()` and uses the returned `conf` directly as the ID-ness score (higher = more ID-like). No negation is applied.

3. **Execution log inconsistency:** Commands 2, 4, 6, 10 produce AUROC values around 86-89% (reasonable for EBO on CIFAR-10 near-OOD), but Command 8 produces AUROC values around 11-14% (near random for a 2-class problem). The only difference between runs is the sign of the scores: Command 8's output shows `"aggregation": "per_dataset_mean_across_runs"` while others show `"dataset_mean_then_run_mean"`, indicating a code change between runs. The `compute_auroc` function negates scores, which is correct when the raw energy score is higher for ID samples. However, the `compute_ebo_scores` function returns `energy` directly (higher = more ID-like), so negating in `compute_auroc` is **wrong** — it makes higher OOD scores correspond to lower AUROC, producing the inverted values seen in Command 8.

4. **The correct approach** (as used by OpenOOD's `Evaluator`): The energy score from `EBOPostprocessor` is already the confidence (higher = more ID-like). For AUROC computation, the OOD score should be the negative of this confidence (or equivalently, the labels should be swapped). The current implementation negates twice (once in `compute_ebo_scores` by returning energy directly, and once in `compute_auroc` by negating), which is correct only when the negation in `compute_auroc` is removed.

**Required repair:** Remove the negation in `compute_auroc`. Change:
```python
scores = np.concatenate([-id_scores, -ood_scores])
```
to:
```python
scores = np.concatenate([id_scores, ood_scores])
```
and swap the label assignment so that ID samples have label 1 and OOD samples have label 0 (since higher energy = more ID-like). Alternatively, keep the negation but swap the labels. The simplest fix is to remove the negation and swap labels:
```python
labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
```

This matches the official OpenOOD evaluation convention and will produce consistent, correct AUROC values across all runs.

REVIEW_STATUS: REPAIR_REQUIRED
