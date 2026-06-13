## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10 (s0/s1/s2)

### Exact Reproduction Command

```bash
# For each seed s0, s1, s2, run:
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Critical:** The `--root` points to the **parent directory** containing s0/, s1/, s2/ subfolders. The evaluator auto-discovers all seeds.

### EBO & AUROC Semantics

- **EBO (Energy-Based OOD):** Score = `temperature * logsumexp(logits / temperature)`. Temperature=1 (default in `scripts/ood/ebo/cifar10_test_ood_ebo.sh`). Higher energy → more OOD-like.
- **AUROC:** Area Under the Receiver Operating Characteristic curve. Reported as **percentage** (0–100). Higher is better for OOD detection.
- **Near-OOD datasets:** CIFAR-100 (label `cifar100`) and TinyImageNet (label `tin`), as defined in `configs/datasets/cifar10/cifar10_ood.yml` lines 14–22.

### Data & Preprocessing

- **ID data:** CIFAR-10 test split (`./data/benchmark_imglist/cifar10/test_cifar10.txt`), images in `./data/images_classic/`.
- **OOD data:** 
  - CIFAR-100: `./data/benchmark_imglist/cifar10/test_cifar100.txt`
  - TinyImageNet: `./data/benchmark_imglist/cifar10/test_tin.txt`
- **Preprocessing:** `base_preprocessor` (CIFAR-10 normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)), image size 32×32, bilinear interpolation.
- **Batch size:** 200 (from `cifar10.yml` test config).

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Seed 0 checkpoint
│   ├── config.yml         # Training config (seed=0)
│   └── log.txt            # Training log (best val acc 94.70%)
├── s1/
│   ├── best.ckpt          # Seed 1 checkpoint
│   └── config.yml         # Training config (seed=1)
└── s2/
    ├── best.ckpt          # Seed 2 checkpoint
    └── config.yml         # Training config (seed=2)
```

Each checkpoint is a `ResNet18_32x32` with 10 output classes, trained with CrossEntropy loss for 100 epochs, learning rate 0.1.

### Aggregation

The `eval_ood.py` script (lines 171–195) automatically:
1. Loads each seed's checkpoint
2. Computes per-seed scores
3. Aggregates results across seeds (mean ± std)
4. Reports per-dataset AUROC (CIFAR-100 and TinyImageNet separately)

**No manual averaging needed.** The script outputs a table with per-dataset metrics.

### CPU & Dependency Risks

- **CPU-only:** Set `CUDA_VISIBLE_DEVICES=""` or run without GPU. The script uses `torch.no_grad()` for inference. Expect ~2–5 minutes per seed on CPU (200 batch size, 10K ID + 10K OOD per dataset).
- **Dependencies:** Requires `torch`, `numpy`, `pickle`, `PyYAML`. All present in the fixed environment.
- **Offline:** No internet access needed. All data and checkpoints are local.
- **Potential issue:** The `--save-score` flag writes pickle files. Ensure write permissions in the results directory.

### Expected Output Format

```
# OOD Detection Performance
# Postprocessor: ebo
# ID Data: cifar10
# Model: ResNet18_32x32
# Seeds: [0, 1, 2]

| Dataset     | AUROC (%)   |
|-------------|-------------|
| cifar100    | XX.XX ± Y.YY|
| tin         | XX.XX ± Y.YY|
```

Report the **mean AUROC** across seeds for each Near-OOD dataset as a percentage (e.g., `84.56`).
