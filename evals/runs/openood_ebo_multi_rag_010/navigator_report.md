# Navigator Handoff for Reproducer

## Exact Source Paths

| Component | Path |
|-----------|------|
| Entry script (single run) | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
| Entry script (multi-run evaluator) | `scripts/eval_ood.py` |
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| EBO config | `configs/postprocessors/ebo.yml` |
| Dataset config (CIFAR-10) | `configs/datasets/cifar10/cifar10.yml` |
| OOD dataset config | `configs/datasets/cifar10/cifar10_ood.yml` |
| Network config | `configs/networks/resnet18_32x32.yml` |
| Pipeline config | `configs/pipelines/test/test_ood.yml` |
| Preprocessor config | `configs/preprocessors/base_preprocessor.yml` |
| Evaluation API datasets | `openood/evaluation_api/datasets.py` |
| Evaluation API preprocessor | `openood/evaluation_api/preprocessor.py` |
| Metrics computation | `openood/evaluators/metrics.py` |
| OOD evaluator | `openood/evaluators/ood_evaluator.py` |
| Network loading | `openood/networks/utils.py` |
| Config setup | `openood/utils/config.py` |

## EBO Formula

From `openood/postprocessors/ebo_postprocessor.py`:

```python
output = net(data)
score = torch.softmax(output, dim=1)
_, pred = torch.max(score, dim=1)
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

- **Energy score** = `temperature * logsumexp(logits / temperature, dim=1)`
- **Prediction** = argmax of softmax scores
- **Temperature** defaults to `1` (from `configs/postprocessors/ebo.yml`)

## CIFAR-10 / Near-OOD Data and Preprocessing

### Data sources (from `openood/evaluation_api/datasets.py`)

| Split | Dataset | Imglist path | Data dir |
|-------|---------|-------------|----------|
| ID test | CIFAR-10 | `benchmark_imglist/cifar10/test_cifar10.txt` | `images_classic/` |
| Near-OOD | CIFAR-100 | `benchmark_imglist/cifar10/test_cifar100.txt` | `images_classic/` |
| Near-OOD | TinyImageNet | `benchmark_imglist/cifar10/test_tin.txt` | `images_classic/` |

### Preprocessing (from `openood/evaluation_api/preprocessor.py`)

```python
TestStandardPreProcessor:
  - Convert('RGB')
  - Resize(32, interpolation=BILINEAR)
  - CenterCrop(32)
  - ToTensor()
  - Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
```

- **Same preprocessing** applied to both ID and OOD data
- **No data augmentation** during evaluation

## Checkpoint / Model Loading

From `scripts/ood/ebo/cifar10_test_ood_ebo.sh`:

```bash
--network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
```

- **Network**: ResNet-18 for 32×32 inputs (`resnet18_32x32`)
- **Checkpoint path**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`
- **Loading mechanism**: `openood/networks/utils.py` → `get_network(network_config)` reads checkpoint from `config.network.checkpoint`

## AUROC and Aggregation Semantics

From `openood/evaluators/metrics.py`:

- `compute_all_metrics(conf, label, pred)` computes AUROC, FPR@95, AUPR-IN, AUPR-OUT
- **AUROC**: Area Under the Receiver Operating Characteristic curve
- **Aggregation**: For multiple runs, use `scripts/eval_ood.py` with `--save-score --save-csv` to save per-sample scores, then aggregate across seeds

From `scripts/eval_ood.py` (recommended multi-run approach):

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

## CPU / Dependency Risks

- **CUDA calls**: `openood/evaluators/ood_evaluator.py` and `openood/postprocessors/base_postprocessor.py` use `torch.no_grad()` but may default to CUDA if available
- **CPU fallback**: Set `CUDA_VISIBLE_DEVICES=""` or use `--num_gpus 0` in config
- **Dependencies**: PyTorch, torchvision, PyYAML, gdown (for dataset downloads)
- **Data download**: `openood/evaluation_api/datasets.py` uses `gdown` for Google Drive downloads

## Minimal Implementation Plan

1. **Setup environment**: Install dependencies (PyTorch, torchvision, PyYAML, gdown)
2. **Download data**: Run `openood/evaluation_api/datasets.py` or manually place CIFAR-10, CIFAR-100, TinyImageNet in `./data/images_classic/` with imglist files in `./data/benchmark_imglist/cifar10/`
3. **Get checkpoint**: Download `cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt` to `./results/`
4. **Single run**: Execute `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (first `main.py` command)
5. **Multi-run aggregation**: Execute `scripts/eval_ood.py` command from same script
6. **Verify**: Check output logs for AUROC on near-OOD (CIFAR-100, TinyImageNet) and far-OOD datasets
