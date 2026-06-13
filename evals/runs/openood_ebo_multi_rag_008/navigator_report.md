# Navigator Handoff: CIFAR-10 EBO OOD Evaluation

## Exact Source Paths

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
| Metrics computation | `openood/evaluators/metrics.py` |
| OOD evaluator | `openood/evaluators/ood_evaluator.py` |
| Config utility | `openood/utils/config.py` |

## EBO Formula

From `openood/postprocessors/ebo_postprocessor.py`:

```python
score = torch.softmax(output, dim=1)
_, pred = torch.max(score, dim=1)
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

- **Energy score** = `temperature * logsumexp(logits / temperature, dim=1)`
- Default temperature = 1 (from `configs/postprocessors/ebo.yml`)
- Sweep parameter: `temperature_list: [1]`

## CIFAR-10 / Near-OOD Data & Preprocessing

### CIFAR-10 (ID)
- **Normalization**: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
- **Image size**: 32×32 (pre_size=32, img_size=32)
- **Test preprocessing** (from `openood/evaluation_api/preprocessor.py`):
  1. Convert to RGB
  2. Resize to 32×32 (bilinear interpolation)
  3. CenterCrop to 32×32
  4. ToTensor
  5. Normalize with CIFAR-10 stats

### Near-OOD Datasets (from `openood/evaluation_api/datasets.py`)
- **CIFAR-100**: `benchmark_imglist/cifar10/test_cifar100.txt`
- **TinyImageNet (tin)**: `benchmark_imglist/cifar10/test_tin.txt`
- Both use same preprocessing as ID data (CIFAR-10 normalization)

### Far-OOD Datasets
- MNIST, SVHN, Texture, Places365

## Checkpoint / Model Loading

### Checkpoint path (from `cifar10_test_ood_ebo.sh`):
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
```

### Model architecture: `ResNet18_32x32` (from `openood/networks/`)

### Loading mechanism (from `scripts/eval_ood.py`):
```python
model_arch = ResNet18_32x32
net = model_arch(num_classes=num_classes)
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
```

## AUROC & Aggregation Semantics

### Metrics computed (from `openood/evaluators/metrics.py`):
- AUROC, AUPR, FPR@95, Detection Error
- Computed per OOD dataset (CIFAR-100, TinyImageNet, etc.)

### Aggregation (from `scripts/eval_ood.py`):
```python
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)
```
- Multiple runs (seeds) are stacked and averaged
- Output format: `"mean ± std"` per metric per OOD dataset

### Running multiple seeds:
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
- This scans subfolders (s0, s1, ...) for checkpoints

## CPU / Dependency Risks

### CUDA calls in codebase:
- `openood/postprocessors/temp_scaling_postprocessor.py` has hardcoded `.cuda()` calls
- `openood/evaluators/ood_evaluator.py` may have CUDA assumptions
- `openood/pipelines/utils.py` may have device handling

### CPU compatibility:
- The `scripts/eval_ood.py` evaluator uses `map_location` for checkpoint loading
- Need to ensure all `.cuda()` calls are replaced or guarded
- The `EBOPostprocessor` itself is device-agnostic (uses `torch.no_grad()`)

### Key dependencies:
- PyTorch, torchvision
- NumPy, pandas
- OpenOOD package structure

## Minimal Implementation Plan

1. **Setup environment**: Install OpenOOD dependencies, ensure `PYTHONPATH='.'` is set

2. **Data preparation**: 
   - Download CIFAR-10, CIFAR-100, TinyImageNet datasets
   - Place in `./data/images_classic/` with benchmark imglist files

3. **Model checkpoint**: 
   - Obtain `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`
   - Or train from scratch using OpenOOD training scripts

4. **Single run evaluation**:
   ```bash
   python main.py \
       --config configs/datasets/cifar10/cifar10.yml \
       configs/datasets/cifar10/cifar10_ood.yml \
       configs/networks/resnet18_32x32.yml \
       configs/pipelines/test/test_ood.yml \
       configs/preprocessors/base_preprocessor.yml \
       configs/postprocessors/ebo.yml \
       --num_workers 8 \
       --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
       --mark 1 \
       --postprocessor.postprocessor_args.temperature 1
   ```

5. **Multi-run aggregation**:
   ```bash
   python scripts/eval_ood.py \
       --id-data cifar10 \
       --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
       --postprocessor ebo \
       --save-score --save-csv
   ```

6. **CPU adaptation**: Replace `.cuda()` calls with device-agnostic code (use `torch.device('cpu')` or `next(net.parameters()).device`)

7. **Verify**: Compare AUROC values against expected ranges (do not reference private published values)
