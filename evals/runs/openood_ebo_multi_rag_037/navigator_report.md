## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Source Paths

| Component | Path |
|-----------|------|
| Eval script (legacy) | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
| Unified evaluator | `scripts/eval_ood.py` |
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| Metrics computation | `openood/evaluators/metrics.py` |
| OOD evaluator | `openood/evaluators/ood_evaluator.py` |
| Evaluation API | `openood/evaluation_api/evaluator.py` |
| OOD config | `configs/datasets/cifar10/cifar10_ood.yml` |
| Checkpoints root | `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt` |

### EBO and AUROC Semantics

**EBO (Energy-Based OOD)**: The postprocessor computes `score = temperature * logsumexp(logits / temperature)` as the OOD confidence. ID samples produce higher energy scores; OOD samples produce lower scores. Temperature is fixed at 1.0 (default in `ebo.yml` and the shell script).

**AUROC**: Computed in `openood/evaluators/metrics.py` via `sklearn.metrics.roc_curve`. The convention treats OOD as the positive class (`ood_indicator = 1` for label == -1). Since EBO produces higher scores for ID, the code negates the confidence scores (`-conf`) before ROC analysis. The returned AUROC is a fraction (0-1), then multiplied by 100 in the evaluator output.

### Data and Preprocessing

- **ID data**: CIFAR-10 test set (10,000 images, 32×32)
- **Near-OOD datasets** (from `cifar10_ood.yml`):
  - CIFAR-100: `data/benchmark_imglist/cifar10/test_cifar100.txt`
  - TinyImageNet (tin): `data/benchmark_imglist/cifar10/test_tin.txt`
- **Data directory**: `./data/images_classic/`
- **Preprocessing**: Base preprocessor (`configs/preprocessors/base_preprocessor.yml`) with CIFAR-10 normalization, 32×32 images, bilinear interpolation
- **Batch size**: 128 for OOD, 200 for ID test (from configs)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # ResNet-18_32x32, 10 classes, seed 0
│   ├── config.yml
│   └── log.txt            # Val Acc 94.70%, Test Acc 95.21%
├── s1/
│   └── best.ckpt
└── s2/
    └── best.ckpt
```

### Aggregation

The recommended approach uses `scripts/eval_ood.py` with `--root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default --postprocessor ebo`. This script:
1. Iterates over `s0`, `s1`, `s2` subfolders
2. Loads each `best.ckpt` with `map_location='cpu'`
3. Runs EBO evaluation per seed
4. Aggregates metrics (mean ± std across seeds)
5. Reports percentage AUROC

### CPU and Dependency Risks

- **CPU-only**: The `eval_ood.py` script calls `net.cuda()` at line 148 — **must be removed/changed to `cpu()`**. The `main.py` pipeline also assumes GPU.
- **Offline**: All data and checkpoints are local; no downloads needed.
- **Dependencies**: Requires `torch`, `numpy`, `sklearn`, `pickle`, `tqdm`. All should be available in the fixed environment.
- **PYTHONPATH**: Must include the repository root (e.g., `PYTHONPATH='.':$PYTHONPATH`).

### Execution Command (CPU-adapted)

```bash
PYTHONPATH='.':$PYTHONPATH python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Critical modification**: In `scripts/eval_ood.py`, change line 148 from `net.cuda()` to `net.cpu()` (or remove the line entirely since CPU is default). The `Evaluator` class in `openood/evaluation_api/evaluator.py` does not call `.cuda()` internally.

### Expected Output Format

The script prints per-seed and aggregated metrics. The Near-OOD AUROC values (percentage) for CIFAR-100 and TinyImageNet will appear under the "nearood" section, reported as mean ± std across the three seeds.
