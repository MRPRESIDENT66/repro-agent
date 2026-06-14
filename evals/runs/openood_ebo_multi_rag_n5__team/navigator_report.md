## Grounded Handoff

### Exact Source Paths

- **Evaluation script**: `scripts/eval_ood.py` (lines 39-55, 171-178)
- **EBO postprocessor**: `openood/postprocessors/ebo_postprocessor.py` (full class)
- **Checkpoints**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/config.yml`
- **Data configs**: `configs/datasets/cifar10/cifar10.yml`, `configs/datasets/cifar10/cifar10_ood.yml`
- **Network config**: `configs/networks/resnet18_32x32.yml`
- **Preprocessor config**: `configs/preprocessors/base_preprocessor.yml`
- **Postprocessor config**: `configs/postprocessors/ebo.yml`
- **Metrics**: `openood/evaluators/metrics.py` (via `compute_all_metrics`)
- **Evaluation API**: `openood/evaluation_api/evaluator.py` (class `Evaluator`)

### EBO and AUROC Semantics

**EBO (Energy-Based OOD)**: The postprocessor computes the energy score as `temperature * logsumexp(logits / temperature, dim=1)`. From `ebo_postprocessor.py` lines 20-22:
```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```
The default temperature is 1.0 (set in `cifar10_test_ood_ebo.sh` line 28). Higher energy → more OOD-like.

**AUROC**: Computed by `compute_all_metrics` in `openood/evaluators/metrics.py`. The metric treats the energy score as the OOD detection score (higher = more OOD). AUROC is reported as a percentage (0-100). The evaluation compares ID (CIFAR-10 test) vs each OOD dataset separately, then averages across Near-OOD datasets.

### Data and Preprocessing

- **ID data**: CIFAR-10 test split from `./data/images_classic/` with imglist `./data/benchmark_imglist/cifar10/test_cifar10.txt`
- **Near-OOD data**: CIFAR-100 and TinyImageNet (configured in `cifar10_ood.yml`)
- **Preprocessing**: `base_preprocessor` (no special augmentations, just normalization)
- **Normalization**: CIFAR-10 specific (`normalization_type: cifar10`)
- **Image size**: 32x32 (bilinear interpolation)
- **Batch size**: 200 (from config and `eval_ood.py` default)

### Checkpoint Layout

Three independent seeds under `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/`:
- `s0/best.ckpt` (seed 0, best validation accuracy 94.70%, test accuracy 95.21%)
- `s1/best.ckpt` (seed 1)
- `s2/best.ckpt` (seed 2)

Each checkpoint is a standard PyTorch state dict for `ResNet18_32x32` (10-class output).

### Aggregation

The official evaluation script `scripts/eval_ood.py` (lines 171-178) handles multi-seed aggregation automatically when `--root` points to the parent directory containing `s0/`, `s1/`, `s2/`. It:
1. Loads pre-computed scores if available (from `scores.pkl` in each subfolder)
2. Otherwise runs inference for each seed
3. Averages metrics across seeds

The Near-OOD AUROC is the average across CIFAR-100 and TinyImageNet (both Near-OOD datasets).

### CPU/Dependency Risks

1. **CPU-only**: The codebase uses `.cuda()` calls in `base_evaluator.py` line 37 (`data = batch['data'].cuda()`). This will crash on CPU. Must either:
   - Set `CUDA_VISIBLE_DEVICES=""` and modify code to use CPU, OR
   - Use `--device cpu` flag if supported (check `eval_ood.py` for device argument)
   - The `Evaluator` class in `evaluation_api/evaluator.py` may handle device differently

2. **Offline**: All data and checkpoints are pre-downloaded. No network calls needed.

3. **Dependencies**: Requires PyTorch, numpy, tqdm, PyYAML, pickle. All should be present in the fixed environment.

### Reproduction Command

```bash
cd /path/to/openood
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Critical**: Ensure CPU compatibility by either:
- Setting environment variable `CUDA_VISIBLE_DEVICES=""` and patching `.cuda()` calls, or
- Using `--device cpu` if the script supports it (check `eval_ood.py` for argparse argument)

The expected output is a percentage AUROC (e.g., ~85-90% range for EBO on CIFAR-10 Near-OOD). The `--save-csv` flag will produce a CSV with per-dataset and average metrics.
