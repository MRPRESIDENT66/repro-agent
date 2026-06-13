# Navigator Handoff: CIFAR-10 EBO OOD Evaluation

## Source Paths

| Component | Path |
|-----------|------|
| Entry script | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
| Unified evaluator | `scripts/eval_ood.py` |
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| EBO config | `configs/postprocessors/ebo.yml` |
| Dataset config (CIFAR-10) | `configs/datasets/cifar10/cifar10.yml` |
| OOD config (CIFAR-10) | `configs/datasets/cifar10/cifar10_ood.yml` |
| Network config | `configs/networks/resnet18_32x32.yml` |
| Pipeline config | `configs/pipelines/test/test_ood.yml` |
| Preprocessor config | `configs/preprocessors/base_preprocessor.yml` |
| Evaluation API datasets | `openood/evaluation_api/datasets.py` |
| Evaluation API preprocessor | `openood/evaluation_api/preprocessor.py` |
| Metrics | `openood/evaluators/metrics.py` |
| OOD evaluator | `openood/evaluators/ood_evaluator.py` |
| Base postprocessor | `openood/postprocessors/base_postprocessor.py` |

## EBO Formula

From `openood/postprocessors/ebo_postprocessor.py`:

```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

Where:
- `output` = logits from the network (shape: [batch, num_classes])
- `temperature` = configurable parameter (default: 1)
- `conf` = energy score (higher = more in-distribution)
- Prediction is taken as `argmax(softmax(output))`

## CIFAR-10 / Near-OOD Data & Preprocessing

### Data sources (from `openood/evaluation_api/datasets.py`):
- **ID**: CIFAR-10 test set (`test_cifar10.txt`)
- **Near-OOD**: CIFAR-100 (`test_cifar100.txt`) and TinyImageNet (`test_tin.txt`)
- **Far-OOD**: MNIST, SVHN, Texture, Places365

### Preprocessing (from `openood/evaluation_api/preprocessor.py`):
```python
# CIFAR-10 specific:
pre_size = 32
img_size = 32
normalization = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]]

# Transform pipeline:
transform = Compose([
    Convert('RGB'),
    Resize(32, interpolation=BILINEAR),
    CenterCrop(32),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])
```

**Important**: All OOD datasets (CIFAR-100, TinyImageNet, etc.) use the **same CIFAR-10 preprocessor** (resize to 32×32, normalize with CIFAR-10 stats). This is critical for reproduction.

## Checkpoint / Model Loading

### Checkpoint path (from `cifar10_test_ood_ebo.sh`):
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
```

### Model architecture:
- `ResNet18_32x32` (from `openood/networks/`)
- 10 output classes (CIFAR-10)
- Loaded via `get_network(config.network)` in `test_acc_pipeline.py`

### For multiple runs (from `scripts/eval_ood.py`):
```python
root = args.root  # e.g., ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default
# Scans subfolders (s0, s1, ...) for checkpoints
```

## AUROC & Aggregation Semantics

### Metrics computed (from `openood/evaluators/metrics.py`):
- AUROC, AUPR (in/out), FPR@95, Detection Error
- Computed per OOD dataset (CIFAR-100, TinyImageNet, etc.)

### Aggregation (from `scripts/eval_ood.py`):
```python
all_metrics = np.stack(all_metrics, axis=0)  # shape: [num_runs, num_ood_datasets, num_metrics]
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)
```

**Key**: Mean and std are computed **across training runs** (seeds), not across OOD datasets. Each OOD dataset gets its own mean±std.

## CPU / Dependency Risks

### CUDA calls in OOD inference:
- `openood/evaluators/ood_evaluator.py` uses `.cuda()` for data/model
- `openood/postprocessors/base_postprocessor.py` assumes CUDA
- `scripts/eval_ood.py` uses `torch.cuda.is_available()` check

**Risk**: Code will fail on CPU-only machines. Need to add `device` parameter or conditional `.cuda()` calls.

### Dependencies:
- PyTorch, torchvision (≥0.13 for `InterpolationMode`)
- numpy, pandas, pickle, yaml
- gdown (for dataset downloads)
- No exotic packages

## Minimal Implementation Plan

1. **Setup environment**: Install PyTorch, torchvision, numpy, pandas, pyyaml
2. **Download data**: Use OpenOOD's benchmark_imglist and images_classic structure, or implement custom dataloaders with the same preprocessing
3. **Load model**: ResNet18_32x32 with 10 classes, load checkpoint from provided path
4. **Implement EBO postprocessor**:
   - Forward pass → logits
   - `conf = temperature * logsumexp(logits / temperature, dim=1)`
   - `pred = argmax(softmax(logits))`
5. **Run evaluation**:
   - For each OOD dataset, compute confidence scores
   - Compute AUROC, FPR@95, AUPR
6. **Aggregate across seeds**: Mean ± std over multiple checkpoint runs
7. **Handle CPU**: Add device parameter to avoid CUDA errors

### Recommended approach:
Use `scripts/eval_ood.py` with:
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This handles multiple runs, aggregation, and CSV output automatically.
