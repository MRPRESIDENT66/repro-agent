## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 28-33)

### Checkpoint Layout

The root directory `./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/` contains three subfolders:
- `s0/` with `best.ckpt` (seed 0, best validation accuracy 94.70% at epoch 96, final test accuracy 95.21%)
- `s1/` with `best.ckpt` (seed 1)
- `s2/` with `best.ckpt` (seed 2)

**Source:** `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/log.txt` (lines 201-207), `s0/config.yml` (line 52), `s1/config.yml` (line 52)

### EBO and AUROC Semantics

- **EBO (Energy-Based OOD Detection):** Uses negative energy score `-E(x) = logsumexp(f(x)/T)` where `T=1` (default temperature). Higher energy → more likely ID.
- **AUROC:** Area Under the Receiver Operating Characteristic curve, treating ID as positive class. Reported as percentage (e.g., 95.23 means 95.23%).
- **Near-OOD datasets:** CIFAR-100 and TinyImageNet (automatically selected by `--id-data cifar10` in `eval_ood.py`).

**Source:** `scripts/eval_ood.py` (lines 40-55), `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (line 24)

### Data and Preprocessing

- **ID data:** CIFAR-10 test split from `./data/images_classic/` with imglist `./data/benchmark_imglist/cifar10/test_cifar10.txt`
- **Preprocessing:** Base preprocessor with bilinear interpolation, resize to 32×32, CIFAR-10 normalization
- **Batch size:** 200 (default in `eval_ood.py`)
- **OOD data:** Automatically loaded by `Evaluator` for Near-OOD benchmarks (CIFAR-100, TinyImageNet)

**Source:** `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` (lines 3-34)

### Aggregation

The `eval_ood.py` script iterates over all `s*` subfolders, computes metrics per seed, then averages across seeds. Final output includes mean and standard deviation for each metric.

**Source:** `scripts/eval_ood.py` (lines 82-84, 146-150)

### CPU/Dependency Risks

- **CPU-only:** The script uses PyTorch which will fall back to CPU automatically. Expect slower evaluation (~5-10 minutes per seed on CPU).
- **Offline:** All data and checkpoints are local. No network calls needed.
- **Dependencies:** Requires `torch`, `numpy`, `scipy`, `pyyaml`, `opencv-python`, `tqdm`, `tensorboard`, `wandb` (wandb may warn but won't fail offline).
- **Potential issue:** `--num_workers 8` in the shell script but `eval_ood.py` uses default batch_size=200. The script handles this internally.

### Expected Output Format

The script prints per-seed and averaged metrics including:
- `AUROC (Near-OOD)` for CIFAR-100 and TinyImageNet
- Standard deviations across seeds

Example output line: `FINAL: AUROC (Near-OOD): 95.23 ± 0.15`

### Verification Steps

1. Run the command above
2. Check that all three seeds (s0, s1, s2) are processed
3. Verify Near-OOD AUROC values are reported as percentages
4. Compare with official OpenOOD benchmark table for EBO on CIFAR-10
