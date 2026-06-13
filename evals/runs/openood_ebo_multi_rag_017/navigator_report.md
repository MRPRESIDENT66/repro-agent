## Handoff Report: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 22-27)

### Checkpoint Layout

The root directory `./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/` contains three subfolders:
- `s0/` with `best.ckpt` (seed 0)
- `s1/` with `best.ckpt` (seed 1)
- `s2/` with `best.ckpt` (seed 2)

**Source:** `scripts/eval_ood.py` lines 76-84 (iterates over `s*` subfolders)

### EBO and AUROC Semantics

- **EBO (Energy-Based OOD):** Uses energy score `E(x) = -T * logsumexp(f(x)/T)` where `T=1` (default temperature). Higher energy = more OOD-like.
- **AUROC:** Area Under the Receiver Operating Characteristic curve. Reported as percentage (0-100). Higher is better for OOD detection.
- **Near-OOD datasets:** CIFAR-100 and TinyImageNet (both resized to 32x32).

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` line 19 (`--postprocessor.postprocessor_args.temperature 1`)

### Data and Preprocessing

- **ID data:** CIFAR-10 test set (10 classes, 32x32 images)
- **OOD data:** CIFAR-100 and TinyImageNet (near-OOD)
- **Preprocessing:** Base preprocessor (normalization_type: cifar10, image_size: 32, bilinear interpolation)
- **Data location:** `./data/images_classic/` with imglist files in `./data/benchmark_imglist/cifar10/`

**Source:** `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` lines 3-34

### Aggregation

The `scripts/eval_ood.py` script automatically:
1. Iterates over all `s*` subfolders (s0, s1, s2)
2. Computes metrics for each seed independently
3. Reports mean and standard deviation across seeds for each metric

**Source:** `scripts/eval_ood.py` lines 82-84, 146-157

### CPU/Dependency Risks

- **CPU-only:** The script uses `map_location='cpu'` when loading checkpoints (line 146 of `eval_ood.py`). However, line 148 calls `net.cuda()` which will **fail on CPU-only environment**. You must modify line 148 to remove `.cuda()` or add a conditional.
- **Offline:** All data and checkpoints are pre-downloaded. No internet access needed.
- **Dependencies:** Requires PyTorch, numpy, pickle, glob, os, collections, argparse.

### Required Modification for CPU

In `scripts/eval_ood.py`, change line 148 from:
```python
net.cuda()
```
to:
```python
# net.cuda()  # Comment out for CPU-only
```

### Expected Output Format

The script prints per-seed metrics and final aggregated results. The Near-OOD AUROC values will be reported as percentages (e.g., `94.32 ± 0.21`). The `--save-csv` flag also writes results to CSV files in each subfolder.
