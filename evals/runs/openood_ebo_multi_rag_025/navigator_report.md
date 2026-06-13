## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command

Run the following from the repository root (CPU-only, offline):

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 27-32)

### Semantics

- **EBO (Energy-Based OOD Detection):** Uses negative energy score `-E(x) = T * log(sum(exp(f(x)/T)))` as OOD score. Higher energy → more OOD-like. Temperature `T=1` (default in postprocessor config).
- **AUROC:** Area Under the Receiver Operating Characteristic curve. Reported as percentage (e.g., 95.21%).
- **Near-OOD:** Evaluated on CIFAR-100 and TinyImageNet (both resized to 32×32). The evaluator automatically uses these two datasets as near-OOD benchmarks for CIFAR-10 ID.

### Data & Preprocessing

- **ID data:** CIFAR-10 (10 classes, 32×32, normalized with CIFAR-10 stats)
- **OOD data:** CIFAR-100 and TinyImageNet (both 32×32, same normalization)
- **Preprocessor:** `base_preprocessor` (simple normalization, no augmentations)
- **Data paths:** `./data/images_classic/` with imglist files in `./data/benchmark_imglist/cifar10/`
- **Batch size:** 200 (default in `eval_ood.py`)

### Checkpoint Layout

The root directory must contain subfolders `s0/`, `s1/`, `s2/`, each with a `best.ckpt` file:

```
./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt
├── s1/best.ckpt
└── s2/best.ckpt
```

**Source:** `scripts/eval_ood.py` lines 76-80 (expects `s*` subfolders)

### Aggregation

The evaluator (`openood.evaluation_api.Evaluator`) automatically:
1. Iterates over `s0`, `s1`, `s2` subfolders
2. Loads each checkpoint into `ResNet18_32x32(num_classes=10)`
3. Computes per-seed AUROC for each near-OOD dataset
4. Reports **mean ± std** across the 3 seeds for each metric

**Source:** `scripts/eval_ood.py` lines 82-84 (loop over `s*` subfolders)

### CPU & Dependency Risks

- **CPU-only:** The script uses PyTorch which will fall back to CPU automatically. Expect ~30-60 minutes total for all 3 seeds × 2 OOD datasets.
- **Offline:** All data and checkpoints are local. No internet access needed.
- **Dependencies:** Requires `torch`, `torchvision`, `numpy`, `pyyaml`, `glob`. All present in the fixed environment.
- **No GPU flags needed:** The script has no GPU requirement; remove any `--gres` or `srun` wrappers.

### Expected Output Format

The script prints per-seed and aggregated metrics to stdout, including:
- Near-OOD AUROC for CIFAR-100 and TinyImageNet
- Mean ± std across s0, s1, s2

If `--save-csv` is used, results are written to a CSV file in the root directory.

### Verification

To confirm correct execution, check that the output contains lines like:
```
Near-OOD AUROC (CIFAR-100): XX.XX%
Near-OOD AUROC (TinyImageNet): XX.XX%
```

The reported values should match the official OpenOOD benchmark for EBO on CIFAR-10 with ResNet-18.
