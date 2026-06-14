## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 40-84, 113-124, 139-150)
- **Shell example**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 17-33)
- **Checkpoint configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`, `s1/config.yml`, `s2/config.yml`
- **Checkpoint files**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`, `s1/best.ckpt`, `s2/best.ckpt`

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses `postprocessor_name='ebo'` with temperature=1. Energy score = `T * logsumexp(logits / T)`. Higher energy → more OOD-like.
- **AUROC**: Area Under the Receiver Operating Characteristic curve. Reported as percentage (e.g., 94.50 means 94.50%). Computed per OOD dataset then averaged across Near-OOD datasets.
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (both 32×32 resized). These are the two "near" OOD datasets in the OpenOOD CIFAR-10 benchmark.

### Data and Preprocessing
- **ID data**: CIFAR-10 (32×32, bilinear interpolation, CIFAR-10 normalization)
- **Data directories**: `./data/images_classic/` with imglist files at `./data/benchmark_imglist/cifar10/`
- **Preprocessor**: `base_preprocessor.yml` (standard normalization, no augmentations at test time)
- **Batch size**: 200 (default in eval_ood.py)

### Checkpoint Layout
- Root: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/`
- Subfolders: `s0/`, `s1/`, `s2/` (each contains `best.ckpt` and `config.yml`)
- Network: `ResNet18_32x32` with 10 classes, no pretrained weights
- Seeds: s0=0, s1=1, s2=2

### Aggregation
- `scripts/eval_ood.py` iterates over `s*` subfolders, evaluates each checkpoint separately, then averages metrics across seeds.
- Final reported AUROC = mean across s0, s1, s2 for each OOD dataset, then mean across the two Near-OOD datasets.

### CPU/Dependency Risks
- **CPU-only**: The script uses PyTorch; ensure `torch.load(..., map_location='cpu')` is used. The `Evaluator` class in `openood.evaluation_api` handles this automatically.
- **Offline**: All data and checkpoints are local. No internet access needed.
- **Dependencies**: `torch`, `torchvision`, `numpy`, `scipy`, `pyyaml`, `glob`, `collections`. All should be present in the fixed environment.

### Reproduction Command
```bash
cd /path/to/openood
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Expected Output Format
The script prints per-seed and averaged metrics. The Near-OOD AUROC is reported as a percentage (e.g., `Near-OOD AUROC: 94.50%`). The final number to report is the average across s0, s1, s2 for CIFAR-100 and TinyImageNet combined.
