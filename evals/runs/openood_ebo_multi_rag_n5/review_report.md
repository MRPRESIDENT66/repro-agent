## Audit Report

### 1. Execution Failure Analysis

The initial execution failed with `FileNotFoundError: '/data/benchmark_imglist/cifar10/test_cifar10.txt'`. The code constructed the path by joining `os.path.dirname(data_root)` with `benchmark_imglist/cifar10/test_cifar10.txt`, but `data_root` was set to `/workspace/data`, making the parent `/workspace`. The actual data directory is `/workspace/data`, so the correct path should use `data_root` directly, not its parent.

After the fix (Command 6), execution succeeded but produced AUROC values around 12%, far below the 50% random baseline. This indicates a fundamental scoring polarity issue.

### 2. Scoring Polarity Bug

The `compute_auroc` function sorts scores in **descending** order (higher score = more ID-like), but the code passes **negative energy** as the score. The energy score formula is `T * logsumexp(logits / T)`. For in-distribution samples, logits are higher, so energy is higher (less negative). Therefore, **higher energy** (less negative) should indicate ID, and **lower energy** (more negative) should indicate OOD.

The code computes `-energies` as the score, which inverts this: more negative energy becomes a higher score. This is correct in intent (higher score = more ID-like), but the AUROC computation sorts descending, expecting higher scores for ID. The issue is that the energy values themselves are negative, and negating them makes them positive but preserves the ordering: if ID samples have higher energy (e.g., -5 vs -10 for OOD), then `-energy` gives 5 for ID and 10 for OOD, meaning OOD gets higher scores. This is **inverted**.

**Fix**: Use raw energy (not negated) as the score, and sort ascending in AUROC, OR keep negated energy but sort ascending. The simplest fix: use raw energy and sort ascending (lower energy = more OOD-like).

### 3. Dataset Count Mismatch

The public contract audit flags that TinyImageNet should have 7793 samples, but the code reports 9000. This is because the code incorrectly uses `len(id_loader.dataset)` for both datasets, which always gives the CIFAR-10 test set size (9000). The OOD dataset sizes are never tracked. This is a reporting bug, not a data loading bug.

### 4. Per-Dataset AUROC Anomaly Check

The reported per-dataset AUROCs are:
- CIFAR-100: ~13-14%
- TinyImageNet: ~10-11%

Both are similarly low (below chance), which is consistent with the polarity inversion. There is no single dataset with anomalously high AUROC (e.g., 98-100%) that would indicate a missing resize step. The low values are entirely explained by the polarity bug.

### 5. Transform Verification

The `build_test_transform` function includes `Resize(32)` and `CenterCrop(32)`, which matches the OpenOOD test preprocessor for 32×32 images. This is correct for CIFAR-10 and CIFAR-100. For TinyImageNet (which is 64×64), the resize to 32×32 is also correct per OpenOOD conventions. No missing resize step is detected.

### Required Repairs

1. **Fix scoring polarity**: Change the score to raw energy (not negated) and sort ascending in AUROC, OR keep negated energy but sort ascending. The current descending sort with negated energy gives inverted results.

2. **Fix dataset count reporting**: Track actual OOD dataset sizes instead of reusing ID dataset size.

3. **Fix data path construction**: Ensure `data_root` is used correctly (not its parent) for finding benchmark imglist files.

REVIEW_STATUS: REPAIR_REQUIRED
