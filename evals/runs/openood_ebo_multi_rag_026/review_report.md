REVIEW_STATUS: REPAIR_REQUIRED

## Finding: EBO score polarity is inverted, causing AUROC to be near 12% instead of the expected ~87%

**Evidence from repository:**

1. **Official EBO postprocessor** (`openood/postprocessors/ebo_postprocessor.py`, line 22):
   ```python
   conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
   ```
   This score is returned as `conf` (confidence). In the OpenOOD evaluation framework, higher confidence = more ID-like. The official `scripts/eval_ood.py` uses this postprocessor directly via the `Evaluator` API, which correctly interprets higher scores as ID.

2. **Your implementation** (`eval_ebo.py`, function `compute_auroc`):
   ```python
   return roc_auc_score(labels, -scores)
   ```
   You negate the EBO scores before computing AUROC. This is incorrect because EBO scores are already higher for ID (confidence). Negating them makes higher OOD scores appear as ID, producing the observed ~12% AUROC (inverted from ~88%).

3. **Execution log confirms the polarity error**:
   - First run (correct polarity, no negation): `actual: 87.58` — matches expected EBO AUROC range
   - Second run (with negation): `actual: 12.42` — exactly `100 - 87.58`, confirming the inversion

4. **The handoff specification** explicitly states: "AUROC: Higher score = ID; lower = OOD; standard ROC analysis" — yet the code negates scores, violating this contract.

**Fix required:** Remove the negation in `compute_auroc`. Change:
```python
return roc_auc_score(labels, -scores)
```
to:
```python
return roc_auc_score(labels, scores)
```

This single-line change will restore the correct polarity and produce the expected ~87% AUROC.

REVIEW_STATUS: REPAIR_REQUIRED
