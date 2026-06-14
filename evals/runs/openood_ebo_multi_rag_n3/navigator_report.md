## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 40-84, 113-146)
- **Shell example**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 17-33)
- **Checkpoint configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`, `s1/config.yml`, `s2/config.yml`
- **Pipeline**: `openood/pipelines/test_ood_pipeline.py` (full file)

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Postprocessor `ebo` with temperature=1. Energy score = `temperature * logsumexp(logits / temperature)`. Lower energy → ID, higher energy → OOD.
- **AUROC**: Area Under Receiver Operating Characteristic curve, reported as percentage (e.g., 94.50 means 94.50%). Computed per OOD dataset, then averaged across Near-OOD datasets (CIFAR-100 + TinyImageNet) per seed, then averaged across seeds s0/s1/s2.

### Data and Preprocessing
- **ID data**: CIFAR-10 (10 classes, 32×32, normalization_type=cifar10)
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (both 32×32 resized)
- **Data directories**: `./data/images_classic/` with imglist files in `./data/benchmark_imglist/cifar10/`
- **Preprocessor**: `base_preprocessor` (standard normalization)
- **Batch size**: 200 for test/ood

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # seed 0 checkpoint
│   ├── config.yml         # seed 0 config
│   └── log.txt            # training log (best acc 94.70%)
├── s1/
│   ├── best.ckpt          # seed 1 checkpoint
│   └── config.yml         # seed 1 config
└── s2/
    ├── best.ckpt          # seed 2 checkpoint
    └── config.yml         # seed 2 config
```

### Aggregation
1. For each seed (s0, s1, s2): compute AUROC for CIFAR-100 and TinyImageNet separately
2. Average the two AUROC values → per-seed Near-OOD AUROC
3. Average across three seeds → final reported Near-OOD AUROC (percentage)

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or 1 (default 8 in shell script will fail). The `eval_ood.py` script uses `--batch-size 200` which is fine for CPU.
- **Offline**: All data and checkpoints are local. No internet access needed.
- **Dependencies**: Requires PyTorch, torchvision, numpy, scikit-learn (for AUROC). All assumed present in the fixed environment.

### Reproduction Command
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv \
    --num_workers 0
```

The script automatically iterates over s0, s1, s2 subfolders, loads each `best.ckpt`, runs EBO with temperature=1, evaluates on both Near-OOD datasets, and prints per-seed and averaged metrics including AUROC (percentage).
