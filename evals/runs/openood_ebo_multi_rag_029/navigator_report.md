## Handoff Report: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Task Summary
Reproduce official EBO (Energy-Based OOD) Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet datasets. Environment is CPU-only and offline.

### Repository Structure & Key Files

**Evaluation Scripts:**
- `scripts/ood/ebo/cifar10_test_ood_ebo.sh` - Original shell script for single-run evaluation
- `scripts/eval_ood.py` - **Recommended unified evaluator** for multi-run aggregation (lines 146-157, 167-205)

**Config Files:**
- `configs/postprocessors/ebo.yml` - EBO postprocessor config (temperature=1, APS_mode=True)
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` - Seed 0 config (seed=0)
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/config.yml` - Seed 1 config (seed=1)
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/config.yml` - Seed 2 config (seed=2)

**Checkpoint Layout:**
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt
├── s1/best.ckpt
└── s2/best.ckpt
```

### Data & Preprocessing
- **ID Data:** CIFAR-10 (32x32, bilinear interpolation, CIFAR-10 normalization)
- **Near-OOD Datasets:** CIFAR-100, TinyImageNet (via OpenOOD benchmark)
- **Data Root:** `./data/images_classic/`
- **Image Lists:** `./data/benchmark_imglist/cifar10/` (train/val/test)
- **Preprocessor:** `base_preprocessor.yml` (default CIFAR-10 preprocessing)
- **Batch Size:** 200 (eval), 128 (training)

### EBO & AUROC Semantics
- **EBO (Energy-Based OOD):** Uses energy score `E(x) = -T * logsumexp(f(x)/T)` where T=1 (temperature). Higher energy = more OOD-like.
- **AUROC:** Area Under ROC curve; reported as percentage (e.g., 94.50 means 94.50%)
- **Near-OOD:** OOD datasets from similar distribution (CIFAR-100, TinyImageNet)
- **APS_mode:** True (enables additional post-processing)

### Aggregation Method
The `scripts/eval_ood.py` script handles multi-seed aggregation:
1. Loads each seed's checkpoint (`best.ckpt`)
2. Runs EBO evaluation per seed
3. Saves per-seed scores to `scores/ebo.pkl`
4. Aggregates metrics across seeds (mean AUROC)
5. Outputs to console and optionally saves CSV

### CPU & Dependency Risks
1. **CPU-only:** All operations must use `map_location='cpu'` when loading checkpoints (already in `eval_ood.py` line 147)
2. **Offline:** All data and checkpoints must be pre-downloaded
3. **No GPU:** Remove CUDA calls; ensure `net.cuda()` is not called
4. **Dependencies:** torch, numpy, pickle, pandas, PyYAML (all standard in OpenOOD)

### Execution Command
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Expected Output Format
- Console prints per-dataset AUROC (percentage) for CIFAR-100 and TinyImageNet
- CSV saved with aggregated metrics across s0, s1, s2
- Scores cached in `scores/ebo.pkl` per seed directory

### Critical Notes
- Use `--save-score` to cache scores for faster re-runs
- Use `--save-csv` for structured output
- The evaluator automatically handles all 3 seeds in the root directory
- Temperature is fixed at 1 (no sweep needed per config)
- Report AUROC as percentage (e.g., 94.50 not 0.9450)
