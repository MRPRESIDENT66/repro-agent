## Handoff Report: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0/s1/s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet Near-OOD datasets. Environment is CPU-only and offline.

### Repository Structure & Key Paths

**Checkpoints location:**
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt
├── s1/best.ckpt
└── s2/best.ckpt
```

**Evaluation script:** `scripts/eval_ood.py` (preferred over `main.py` for multi-run aggregation)

**Shell reference:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 22-28 show the eval_ood.py usage pattern)

### EBO & AUROC Semantics

- **EBO (Energy-Based OOD Detection):** Uses negative energy score `-E(x) = T * log(Σ exp(f_i(x)/T))` where `f_i` are logits. Higher energy = more OOD-like. Temperature parameter T=1 (default in EBO postprocessor).
- **AUROC:** Area Under ROC curve computed by `openood/evaluators/metrics.py` via `compute_all_metrics()`. Reported as percentage (multiplied by 100 in `ood_evaluator.py` line 142).
- **Near-OOD:** Specifically CIFAR-100 and TinyImageNet datasets, evaluated separately per dataset.

### Data & Preprocessing

- **ID data:** CIFAR-10 (32x32, normalization_type: cifar10)
- **OOD data:** CIFAR-100 and TinyImageNet (automatically handled by `Evaluator` class)
- **Data root:** `./data/` (already present)
- **Preprocessor:** `base_preprocessor` (standard normalization)
- **Batch size:** 200 (default in eval_ood.py)
- **Num workers:** 4 (CPU-safe default)

### Checkpoint Layout

The `eval_ood.py` script expects the following structure under `--root`:
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt
│   └── config.yml
├── s1/
│   ├── best.ckpt
│   └── config.yml
└── s2/
    ├── best.ckpt
    └── config.yml
```

The script iterates over `s*` subfolders, loads each checkpoint, evaluates, and aggregates metrics.

### Aggregation

The `eval_ood.py` script (lines 83-84, 171-195) automatically:
1. Iterates over s0, s1, s2 subfolders
2. Computes per-run metrics (including AUROC per Near-OOD dataset)
3. Aggregates results (mean ± std across runs)
4. Prints final aggregated metrics

### CPU/Dependency Risks

1. **CPU-only:** Set `--num_workers 4` (or lower) to avoid multiprocessing issues. The `Evaluator` class uses `num_workers=4` by default.
2. **Offline:** All data and checkpoints are pre-downloaded. No network calls needed.
3. **Dependencies:** PyTorch, torchvision, numpy, pickle (all standard in OpenOOD environment)

### Execution Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Expected Output Format

The script will print per-run metrics and final aggregated results:
```
[Results] cifar10 - ebo
s0: Near-OOD AUROC: XX.XX% (cifar100), YY.YY% (tinyimagenet)
s1: Near-OOD AUROC: XX.XX% (cifar100), YY.YY% (tinyimagenet)
s2: Near-OOD AUROC: XX.XX% (cifar100), YY.YY% (tinyimagenet)
Mean ± Std: XX.XX ± Z.ZZ (cifar100), YY.YY ± Z.ZZ (tinyimagenet)
```

### Verification Steps

1. Confirm `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt` exist
2. Run the eval_ood.py command above
3. Report the percentage AUROC values for both Near-OOD datasets (CIFAR-100 and TinyImageNet)
