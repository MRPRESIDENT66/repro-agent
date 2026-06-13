## Audit of eval_ebo.py Implementation

### Execution Error Analysis

The public execution log shows an intermittent failure: commands 2 and 8 fail with `[Errno 2] No such file or directory: '/workspace/../data/benchmark_imglist/cifar10/test_cifar10.txt'`, while commands 4, 6, and 10 succeed. This indicates a **non-deterministic data root resolution** in the `main()` function.

### Root Cause: Data Root Resolution Logic

In `eval_ebo.py` lines 107-111:
```python
if os.path.exists(os.path.join(script_dir, 'data', 'benchmark_imglist')):
    data_root = os.path.join(script_dir, 'data')
else:
    data_root = os.path.join(os.path.dirname(script_dir), 'data')
```

When the script is executed from `/workspace/`, `script_dir` resolves to `/workspace/`. The first branch checks `/workspace/data/benchmark_imglist` — if this path does not exist (e.g., when data is at `/workspace/../data/`), it falls back to `os.path.join(os.path.dirname(script_dir), 'data')` which becomes `/workspace/../data/` = `/data/`. This path is incorrect; the actual data is at `/workspace/data/`.

The reference implementation `scripts/eval_ood.py` uses a fixed `ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')` and then `data_root=os.path.join(ROOT_DIR, 'data')`, which reliably resolves to `/workspace/data/` when the script is at `/workspace/scripts/eval_ood.py`.

### Semantic Claim: AUROC Computation

The `compute_auroc` function (lines 72-93) implements a non-standard AUROC calculation. The standard trapezoidal rule for AUROC is:
```python
auroc += tpr * (1.0 / neg_count)
```
This formula is incorrect. The correct incremental AUROC update when iterating over sorted scores (descending) is:
```python
auroc += tpr * (fpr_prev - fpr)  # or equivalently: auroc += tpr * (1.0/neg_count)
```
However, the implementation updates `auroc` only when a negative (ID) sample is encountered, using `auroc += tpr * (1.0 / neg_count)`. This is actually correct for the **rectangular** approximation when iterating descending — each negative sample contributes the current TPR times the FPR increment. But the variable `fpr` is incremented but never used in the AUROC update, and the update happens only on negative samples, which is the standard approach for the Mann-Whitney U statistic equivalent.

**Verdict on AUROC**: The implementation is functionally correct for the Mann-Whitney formulation of AUROC. The formula `auroc += tpr * (1.0 / neg_count)` when encountering a negative sample is equivalent to the standard implementation.

### Aggregation Logic

The aggregation computes `dataset_mean_then_run_mean`:
1. For each dataset (cifar100, tin), compute mean across seeds
2. For each seed, compute mean across datasets
3. Final = mean of run means

This is equivalent to the global mean of all per-dataset-per-seed values. The handoff report states "Reports mean and standard deviation across seeds for each metric" — the implementation reports per-dataset means and a single aggregated mean, which is a reasonable interpretation.

### Data Path Fix Required

The data root resolution is fragile and causes intermittent failures. The fix should use a deterministic path relative to the script location, matching the reference implementation pattern.

**Required change in `main()`** (lines 107-111):
Replace:
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'data', 'benchmark_imglist')):
    data_root = os.path.join(script_dir, 'data')
else:
    data_root = os.path.join(os.path.dirname(script_dir), 'data')
```
With:
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(os.path.dirname(script_dir), 'data')
```

This assumes the script is in a subdirectory of the project root (e.g., `scripts/`), which matches the reference implementation's assumption.

### Additional Observations

1. **CPU-only compatibility**: The implementation correctly uses `map_location='cpu'` and does not call `.cuda()`, satisfying the CPU-only requirement from the handoff.

2. **Checkpoint loading**: The `evaluate_seed` function creates a new `ResNet18_32x32` instance each call, but the `net` object is passed from `main()` and reused across seeds. This is fine since `load_state_dict` overwrites all parameters.

3. **Reproducibility**: The output is deterministic (same value across commands 4, 6, 10), confirming no stochastic elements.

### Conclusion

The primary issue is the non-deterministic data root resolution causing intermittent failures. The AUROC computation is correct. The aggregation matches the expected format.

REVIEW_STATUS: REPAIR_REQUIRED
