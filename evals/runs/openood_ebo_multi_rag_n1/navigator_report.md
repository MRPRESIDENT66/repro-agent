## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Source Paths

| Component | Path |
|-----------|------|
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| Base postprocessor | `openood/postprocessors/base_postprocessor.py` |
| Evaluation pipeline | `openood/pipelines/test_ood_pipeline.py` |
| Unified evaluator | `scripts/eval_ood.py` |
| OOD config | `configs/datasets/cifar10/cifar10_ood.yml` |
| Shell script (reference) | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |

### EBO Semantics

From `ebo_postprocessor.py` lines 17-20:
```python
score = torch.softmax(output, dim=1)
_, pred = torch.max(score, dim=1)
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

EBO score = `temperature * logsumexp(logits / temperature)`. Default temperature = 1.0. Higher score = more confident (ID-like). AUROC treats ID as positive class.

### AUROC Semantics

From `openood/evaluators/` (standard binary classification): AUROC measures separation between ID (CIFAR-10 test) and OOD (CIFAR-100 / TinyImageNet) confidence scores. Reported as percentage (0-100).

### Data and Preprocessing

From `cifar10_ood.yml`:
- **ID test**: CIFAR-10 test set (via `ImglistDataset`, `./data/images_classic/`)
- **Near-OOD datasets**:
  - CIFAR-100: `./data/benchmark_imglist/cifar10/test_cifar100.txt`
  - TinyImageNet: `./data/benchmark_imglist/cifar10/test_tin.txt`
- **Batch size**: 128
- **Preprocessor**: `base_preprocessor.yml` (standard CIFAR-10 normalization)
- **Network**: `resnet18_32x32.yml` (ResNet-18 for 32x32 inputs)

### Checkpoint Layout

Checkpoints at: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`

The `scripts/eval_ood.py` evaluator (lines 79-84) expects this structure:
```python
for subfolder in sorted(glob(os.path.join(root, 's*'))):
```
It auto-discovers `s0`, `s1`, `s2` subfolders.

### Aggregation

The unified evaluator (`scripts/eval_ood.py`) iterates over all `s*` subfolders and aggregates metrics. The exact aggregation method (mean vs. mean±std) should be verified from the evaluator output, but standard practice is mean across seeds.

### CPU/Dependency Risks

1. **GPU dependency in base postprocessor**: `openood/postprocessors/base_postprocessor.py` line 22 has `data = batch['data'].cuda()` — **must be removed/changed to `.cpu()`** for CPU-only execution.
2. **`launch` in main.py**: `main.py` calls `launch()` which expects GPU. Use `scripts/eval_ood.py` instead, which is simpler and doesn't require `launch()`.
3. **Offline**: All data and checkpoints are local; no downloads needed.

### Recommended Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This matches the recommended approach in `cifar10_test_ood_ebo.sh` lines 24-28.

### Expected Output Format

The evaluator prints per-seed and aggregated AUROC for each OOD dataset. Near-OOD results will appear under `cifar100` and `tin` (TinyImageNet) columns. Report as percentage (e.g., `94.32` not `0.9432`).
