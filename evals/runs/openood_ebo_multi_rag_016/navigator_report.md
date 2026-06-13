## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (unified evaluator, recommended over `main.py`)
- **Postprocessor config**: `configs/postprocessors/ebo.yml` (temperature=1, APS_mode=True)
- **Checkpoint root**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/` containing subfolders `s0/`, `s1/`, `s2/` each with `best.ckpt`
- **Network**: `ResNet18_32x32` (from `openood/networks/`)

### EBO & AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses `EBOPostprocessor` from `openood/postprocessors/`. Energy score = `temperature * logsumexp(logits / temperature)`. Higher energy → more OOD-like. Temperature fixed at 1 (no sweep needed per config).
- **AUROC**: Area Under the Receiver Operating Characteristic curve. Reported as percentage (e.g., 95.12 means 95.12%). The `Evaluator.eval_ood()` method computes this automatically.

### Data & Preprocessing
- **ID data**: CIFAR-10 (10 classes, 32×32 images)
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (both 32×32 resized)
- **Preprocessing**: `base_preprocessor.yml` (standard normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) for CIFAR-10)
- **Data location**: Already present in the environment (offline, no download needed)

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   └── best.ckpt
├── s1/
│   └── best.ckpt
└── s2/
    └── best.ckpt
```
Each checkpoint is a PyTorch state dict for `ResNet18_32x32` trained with CrossEntropy loss for 100 epochs.

### Aggregation
The `scripts/eval_ood.py` script automatically:
1. Iterates over all `s*/` subdirectories under `--root`
2. For each run, computes OOD metrics (including AUROC) via `Evaluator.eval_ood()`
3. Aggregates results: **mean ± std** across the 3 runs (s0, s1, s2)
4. Outputs a DataFrame with metrics as rows, OOD datasets as columns

### CPU & Dependency Risks
- **CPU-only**: The script uses PyTorch; ensure `torch.load(..., map_location='cpu')` is used implicitly. The `eval_ood.py` script loads checkpoints with `torch.load(ckpt_path, map_location=device)` where device defaults to CPU if no GPU.
- **Offline**: All data and checkpoints are pre-downloaded. No network calls needed.
- **Dependencies**: Requires `torch`, `numpy`, `pandas`, `pickle`, `glob`. All should be present in the fixed environment.

### Reproduction Command
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
This will print the aggregated metrics table. The Near-OOD AUROC for CIFAR-100 and TinyImageNet will be in the row labeled `AUROC` under columns `CIFAR-100` and `TinyImageNet` (or similar naming). Report as percentage (e.g., `94.50 ± 0.30`).
