# Navigator Handoff for Reproducer

## Exact Source Paths

| Component | Path |
|-----------|------|
| Entry script | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
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
| Unified evaluator | `scripts/eval_ood.py` |
| Sweep helper | `scripts/ood/ebo/sweep_osr.py` |

## EBO Formula

From `openood/postprocessors/ebo_postprocessor.py`:

```python
output = net(data)
score = torch.softmax(output, dim=1)
_, pred = torch.max(score, dim=1)
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

**Key semantics:**
- `conf` is the **energy score** (not softmax confidence)
- Default temperature = 1 (from `configs/postprocessors/ebo.yml`)
- Sweep parameter: `temperature_list: [1]` (single value)
- Higher energy = more in-distribution (ID)
- `pred` is still from softmax argmax (for classification accuracy)

## CIFAR-10 / Near-OOD Data & Preprocessing

### ID Data (CIFAR-10)
- **Train**: `benchmark_imglist/cifar10/train_cifar10.txt`
- **Test**: `benchmark_imglist/cifar10/test_cifar10.txt`
- **Data dir**: `./data/images_classic/`

### Near-OOD Datasets (from `openood/evaluation_api/datasets.py`)
- **CIFAR-100**: `benchmark_imglist/cifar10/test_cifar100.txt`
- **TinyImageNet (tin)**: `benchmark_imglist/cifar10/test_tin.txt`

### Far-OOD Datasets
- MNIST, SVHN, Texture, Places365

### Preprocessing (from `openood/evaluation_api/preprocessor.py`)
```python
TestStandardPreProcessor:
  - Convert('RGB')
  - Resize(32, interpolation=BILINEAR)
  - CenterCrop(32)
  - ToTensor()
  - Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
```

**Critical**: Both ID and OOD data use the **same preprocessor** (CIFAR-10 normalization). OOD images are resized/cropped to 32×32.

## Checkpoint / Model Loading

### Default checkpoint path (from `cifar10_test_ood_ebo.sh`):
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
```

### Alternative checkpoint (from `sweep_posthoc.py`):
```
./results/checkpoints/cifar10_res18_acc95.24.ckpt
```

### Model architecture:
- `resnet18_32x32` (from `configs/networks/resnet18_32x32.yml`)
- 10 output classes

### Loading mechanism:
- `openood/networks/utils.py` → `get_network(network_config)`
- Checkpoint specified via `--network.checkpoint` CLI arg
- Config merging: `setup_config()` in `openood/utils/config.py` merges YAML files then applies CLI overrides

## AUROC & Aggregation Semantics

### Metric computation (`openood/evaluators/metrics.py`):
- `compute_all_metrics(conf, label, pred)` returns dict with `auroc`, `aupr`, `fpr`, etc.
- **AUROC**: Area Under Receiver Operating Characteristic curve
  - ID samples have higher energy scores → treated as positive class
  - OOD samples have lower energy scores → treated as negative class

### Aggregation for multiple runs:
- **Recommended**: Use `scripts/eval_ood.py` with `--save-score --save-csv`
- This saves per-sample scores, enabling aggregation across seeds
- The unified evaluator handles multiple checkpoint directories under `--root`

### Per-dataset metrics:
- Near-OOD: separate AUROC for CIFAR-100 and TinyImageNet
- Far-OOD: separate AUROC for MNIST, SVHN, Texture, Places365
- **No averaging across OOD datasets** in the standard pipeline (each reported separately)

## CPU / Dependency Risks

### CUDA calls in inference:
- `openood/evaluators/ood_evaluator.py` and `base_postprocessor.py` use `torch.no_grad()` but assume CUDA
- `to_np(x)` in `base_evaluator.py` handles CPU tensors but may fail if GPU tensors are expected
- **Risk**: The pipeline may crash on CPU-only machines if `.cuda()` calls are present in network loading

### Workaround for CPU:
- Modify `scripts/eval_ood.py` or `main.py` to add `--device cpu` or set `torch.device('cpu')`
- Ensure checkpoint is loaded with `map_location='cpu'`

### Dependencies:
- PyTorch, torchvision, PyYAML, gdown (for dataset downloads)
- OpenOOD package structure requires `PYTHONPATH='.'` (as shown in all scripts)

## Minimal Implementation Plan

1. **Setup environment**:
   ```bash
   git clone <openood-repo>
   cd openood
   export PYTHONPATH='.':$PYTHONPATH
   pip install -r requirements.txt
   ```

2. **Download data** (if not present):
   - CIFAR-10, CIFAR-100, TinyImageNet → `./data/images_classic/`
   - Imglist files → `./data/benchmark_imglist/`

3. **Get checkpoint**:
   - Either train: `python main.py --config configs/datasets/cifar10/cifar10.yml configs/networks/resnet18_32x32.yml configs/pipelines/train/train.yml configs/preprocessors/base_preprocessor.yml`
   - Or download pre-trained checkpoint to `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`

4. **Run single evaluation**:
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
       --postprocessor.postprocessor_args.temperature 1
   ```

5. **Run multi-seed evaluation** (recommended):
   ```bash
   python scripts/eval_ood.py \
       --id-data cifar10 \
       --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
       --postprocessor ebo \
       --save-score --save-csv
   ```
   - For multiple seeds, point `--root` to parent directory containing `s0/`, `s1/`, etc.

6. **Extract metrics**:
   - Parse stdout for per-dataset AUROC values
   - Or use saved CSV files for custom aggregation
   - Report AUROC separately for each OOD dataset (CIFAR-100, TinyImageNet, MNIST, SVHN, Texture, Places365)

7. **CPU fallback** (if needed):
   - Add `--device cpu` to command line
   - Or modify `scripts/eval_ood.py` to accept device argument
