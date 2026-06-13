REVIEW_STATUS: REPAIR_REQUIRED

## Finding

The execution log shows two different runs of `eval_ebo.py` producing dramatically different AUROC values:

- **Command 2 (first run):** `actual: 12.42%` (near random for AUROC, indicating inverted scores)
- **Command 4 (second run):** `actual: 87.58%` (reasonable near-OOD AUROC)

The code is **non-deterministic** — it produces different results on repeated runs with identical inputs. This violates the fundamental requirement of a reproducible evaluation.

## Root Cause

The `compute_auroc` function in `eval_ebo.py` negates the confidence scores before calling `roc_curve`:

```python
fpr_list, tpr_list, _ = metrics.roc_curve(ood_indicator, -conf)
```

This is correct per the official OpenOOD `metrics.py` (line 56: `metrics.roc_curve(ood_indicator, -conf)`). However, the EBO postprocessor returns **energy scores** where higher energy = more OOD-like. The official `compute_all_metrics` in `metrics.py` expects confidence scores where higher = more ID-like, hence the negation.

The problem is that `compute_auroc` is called with `id_scores` and `ood_scores` that are **already energy scores** (higher = OOD). The negation `-conf` then inverts the relationship, making the AUROC computation dependent on the **order of concatenation** in `compute_auroc`:

```python
ood_indicator = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
conf = np.concatenate([id_scores, ood_scores])
```

The negation `-conf` flips the scores, but the `ood_indicator` is 1 for OOD samples. After negation, lower (more negative) energy becomes "more ID-like" in the ROC curve. This is correct only if the energy scores are **always positive** and the negation preserves the relative ordering. But energy scores from `logsumexp` can be negative, and the negation changes the ranking.

The non-determinism likely comes from **data loading order** (shuffle buffer, multiprocessing) or **floating-point accumulation order** in the energy computation, which changes the exact score values and interacts with the negation to produce different AUROC results.

## Repository Evidence

From `openood/evaluators/metrics.py` (lines 48-56):
```python
def auc_and_fpr_recall(conf, label, tpr_th):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
```

The official code negates `conf` because it assumes `conf` is a **confidence score** (higher = ID). The `eval_ebo.py` code passes raw energy scores (higher = OOD) to `compute_auroc`, which then negates them. This double-negation is incorrect — the energy scores should be passed **without negation** to `roc_curve`, or the `ood_indicator` should be inverted.

## Required Fix

The `compute_auroc` function should **not** negate the confidence scores when using EBO energy scores, because energy scores already have the correct polarity (higher = OOD). The fix is to change:

```python
fpr_list, tpr_list, _ = metrics.roc_curve(ood_indicator, -conf)
```

to:

```python
fpr_list, tpr_list, _ = metrics.roc_curve(ood_indicator, conf)
```

This matches the official OpenOOD convention where `roc_curve` expects higher scores = more OOD-like when `ood_indicator=1` for OOD samples. The energy scores from EBO already satisfy this.

REVIEW_STATUS: REPAIR_REQUIRED
