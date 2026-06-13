## Handoff Report: EBO Near-OOD AUROC for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0/s1/s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet (Near-OOD). CPU-only, offline, fixed OpenOOD repo.

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py`
- **EBO postprocessor**: `openood/postprocessors/ebo_postprocessor.py`
- **Checkpoints**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Data config (eval API)**: `openood/evaluation_api/datasets.py` (lines 46-61 define CIFAR-10 Near-OOD as `cifar100` and `tin`)
- **Metrics**: `openood/evaluators/metrics.py` (via `compute_all_metrics`)

### EBO Semantics
- **Score**: `temperature * logsumexp(logits / temperature, dim=1)` (higher = more ID)
- **Temperature**: 1 (fixed, no sweep needed)
- **AUROC**: Higher score = ID; lower = OOD; standard ROC analysis

### Data & Preprocessing
- **ID**: CIFAR-10 test set (`test_cifar10.txt`)
- **Near-OOD**: CIFAR-100 (`test_cifar100.txt`) and TinyImageNet (`test_tin.txt`)
- **Preprocessor**: `base_preprocessor` (normalize per CIFAR-10 stats, resize to 32×32 bilinear)
- **Data root**: `./data/images_classic/`
- **Imglist root**: `./data/benchmark_imglist/cifar10/`

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt   (seed 0, test acc 95.21)
├── s1/best.ckpt   (seed 1)
└── s2/best.ckpt   (seed 2)
```

### Aggregation
- Run `scripts/eval_ood.py` **once per seed** with `--root` pointing to the parent directory containing all three subdirectories
- The evaluator auto-discovers `s0/`, `s1/`, `s2/` and reports **mean ± std** across seeds
- **Command per seed** (run 3 times, changing `--root` to each seed's folder OR use the parent root for auto-aggregation):
```bash
PYTHONPATH='.' python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
- The evaluator returns per-dataset AUROC; report as **percentage** (multiply by 100)

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or `1` to avoid multiprocessing hangs; the evaluator defaults to `num_workers=4` which may fail on CPU-only
- **Offline**: All data and checkpoints are local; no downloads needed
- **PyTorch**: Must be installed; no GPU required but inference will be slow (~30-60 min total for 3 seeds × 2 OOD datasets)
- **Memory**: ~2GB RAM sufficient for batch_size=200

### Expected Output Format
```
# Near-OOD results (mean ± std across s0,s1,s2):
# CIFAR-100:  XX.XX ± Y.YY
# TinyImageNet: ZZ.ZZ ± W.WW
```

### Critical Implementation Notes
1. The `scripts/eval_ood.py` script auto-detects seed subdirectories under `--root` and aggregates results
2. EBO temperature is hardcoded to 1 in the postprocessor; no sweep needed
3. The evaluator computes AUROC via `compute_all_metrics` which uses `sklearn.metrics.roc_auc_score`
4. Report as percentage (e.g., 85.23 not 0.8523)
