## Handoff Report: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet Near-OOD datasets in CPU-only, offline environment.

### Repository Semantics & Execution Path

**Primary evaluation script**: `scripts/eval_ood.py` (recommended by repository for multi-run results)

**Command template** (from `scripts/ood/ebo/cifar10_test_ood_ebo.sh` lines 27-32):
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### EBO Semantics
From `openood/postprocessors/ebo_postprocessor.py`:
- **Energy score**: `conf = temperature * torch.logsumexp(output / temperature, dim=1)`
- **Temperature**: 1 (default, set via `--postprocessor.postprocessor_args.temperature 1`)
- **Prediction**: argmax of softmax output
- **AUROC**: Computed by `compute_all_metrics` in `openood/evaluators/metrics.py`, reported as percentage (multiplied by 100 in `ood_evaluator.py` line 143)

### Data & Preprocessing
From `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`:
- **ID data**: CIFAR-10 (10 classes, 32x32 images)
- **Data directory**: `./data/images_classic/`
- **Image lists**: `./data/benchmark_imglist/cifar10/` (train/val/test)
- **Preprocessor**: `base_preprocessor.yml` (normalization_type: cifar10)
- **Batch size**: 200 for test
- **Num workers**: 4 (config) / 8 (script override)

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Best validation checkpoint (epoch 96, 94.70% val acc)
│   ├── config.yml         # Seed 0 configuration
│   └── log.txt            # Training log (final test acc 95.21%)
├── s1/
│   ├── best.ckpt          # Seed 1 checkpoint
│   └── config.yml         # Seed 1 configuration
└── s2/
    ├── best.ckpt          # Seed 2 checkpoint
    └── config.yml         # Seed 2 configuration
```

### Aggregation
The `scripts/eval_ood.py` script automatically:
1. Scans all subdirectories (s0, s1, s2) under `--root`
2. Loads each checkpoint and computes per-seed metrics
3. **Averages AUROC across seeds** (implemented in `Evaluator` class)
4. Saves per-seed scores if `--save-score` flag is used
5. Saves CSV if `--save-csv` flag is used

### CPU & Dependency Risks
1. **CPU-only**: The script uses `--num_workers` which spawns multiprocessing workers. Set `--num_workers 0` or `--num_workers 1` to avoid multiprocessing issues in constrained environments.
2. **Offline**: All data and checkpoints are local. No external downloads needed.
3. **Dependencies**: Requires PyTorch, numpy, pickle (for score loading/saving), tqdm
4. **Memory**: Each CIFAR-10 batch of 200 images at 32x32 is ~0.6MB. Total memory should be <2GB for inference.

### Execution Command (CPU-safe)
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv \
    --num_workers 0
```

### Expected Output
- Per-seed AUROC for CIFAR-100 and TinyImageNet (Near-OOD)
- Averaged AUROC across s0, s1, s2
- Reported as percentage (e.g., 85.23%)
- Scores saved to `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/` subdirectories
- CSV output with aggregated metrics

### Verification Points
- Temperature = 1 (default EBO parameter)
- Energy score = `temperature * logsumexp(logits / temperature)`
- AUROC computed via `compute_all_metrics` with OOD labels = -1
- Results are percentages (multiplied by 100)
