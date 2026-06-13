## Handoff: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py`
- **EBO postprocessor**: `openood/postprocessors/ebo_postprocessor.py`
- **Checkpoints**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/config.yml`
- **Metrics**: `openood/evaluators/metrics.py` (via `compute_all_metrics`)
- **Evaluation API**: `openood/evaluation_api/evaluator.py`

### EBO & AUROC Semantics
- **EBO score** = `temperature * logsumexp(logits / temperature, dim=1)` (from `ebo_postprocessor.py` line 22). Default temperature = 1.
- **AUROC**: Area Under the Receiver Operating Characteristic curve, computed by `compute_all_metrics` in `openood/evaluators/metrics.py`. Reported as percentage (0-100).
- **Near-OOD**: CIFAR-100 and TinyImageNet (the two near-OOD datasets for CIFAR-10 in OpenOOD's benchmark).

### Data & Preprocessing
- **ID data**: CIFAR-10 (32×32, normalization_type: cifar10, bilinear interpolation)
- **OOD data**: CIFAR-100 and TinyImageNet (both resized to 32×32 with same preprocessing)
- **Data root**: `./data/images_classic/`
- **Imglist paths**: `./data/benchmark_imglist/cifar10/` (train/val/test)
- **Batch size**: 200 (default in eval_ood.py)
- **Preprocessor**: `base_preprocessor` (standard normalization)

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
- **Per-seed**: Run `scripts/eval_ood.py` separately for each seed (s0, s1, s2) using `--root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default` and `--postprocessor ebo`.
- **Final**: Average the three AUROC values (one per seed) for each Near-OOD dataset. Report as mean percentage.

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or `--num_workers 1` to avoid multiprocessing issues. The `eval_ood.py` script uses `num_workers=4` by default; override with `--num_workers 0`.
- **Offline**: All data and checkpoints are local. No network calls needed.
- **PyTorch**: Must be installed with CPU support only. No CUDA required.
- **Memory**: Each evaluation processes ~10K ID + ~10K OOD images at batch_size=200. Expect ~2-4 GB RAM per run.

### Reproduction Command (per seed)
```bash
PYTHONPATH='.' python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --num_workers 0 \
    --save-score --save-csv
```
This evaluates all three seeds (s0, s1, s2) in one pass and outputs per-seed AUROC for both Near-OOD datasets. The `--save-csv` flag produces a CSV with metrics including AUROC.

### Expected Output Format
The script prints metrics to stdout and saves a CSV. Look for:
- `Near-OOD AUROC` for CIFAR-100 and TinyImageNet
- Values are percentages (e.g., 85.23 means 85.23%)

### Verification
- The official EBO Near-OOD AUROC for CIFAR-10 with ResNet-18 should be in the range ~85-90% for CIFAR-100 and ~88-92% for TinyImageNet (temperature=1).
- If results differ significantly, check that `temperature=1` is the default in `ebo_postprocessor.py` and that no hyperparameter sweep is triggered.
