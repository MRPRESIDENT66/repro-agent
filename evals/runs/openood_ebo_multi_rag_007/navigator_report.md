# Navigator Handoff for Reproducer

## Task
Reproduce CIFAR-10 EBO OOD evaluation using OpenOOD v0.5.1, including near-OOD (CIFAR-100, TinyImageNet) and far-OOD (MNIST, SVHN, Texture, Places365) benchmarks, with AUROC aggregation across multiple training runs.

---

## Exact Source Paths

| Component | Path |
|-----------|------|
| Entry script (single run) | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
| Entry script (multi-run) | `scripts/eval_ood.py` |
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| EBO config | `configs/postprocessors/ebo.yml` |
| Dataset definitions | `openood/evaluation_api/datasets.py` |
| Preprocessing | `openood/evaluation_api/preprocessor.py` |
| Metrics | `openood/evaluators/metrics.py` |
| Network loading | `openood/networks/utils.py` |
| Config parsing | `openood/utils/config.py` |

---

## EBO Formula

From `openood/postprocessors/ebo_postprocessor.py`:

```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```

Where:
- `output` = logits from the network (shape `[batch, num_classes]`)
- `temperature` = hyperparameter (default 1.0, configurable via `postprocessor_args.temperature`)
- `conf` = energy score (higher = more in-distribution)
- Prediction is standard argmax of softmax

---

## CIFAR-10 / Near-OOD Data & Preprocessing

### Dataset structure (from `openood/evaluation_api/datasets.py`)

| Split | Datasets |
|-------|----------|
| ID test | CIFAR-10 |
| Near-OOD | CIFAR-100, TinyImageNet (TIN) |
| Far-OOD | MNIST, SVHN, Texture, Places365 |

### Preprocessing (from `openood/evaluation_api/preprocessor.py`)

All test images use `TestStandardPreProcessor`:
1. Convert to RGB
2. Resize to 32×32 (bilinear interpolation)
3. CenterCrop to 32×32
4. ToTensor
5. Normalize with CIFAR-10 stats: `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`

**Important**: All OOD datasets (including MNIST, SVHN) are converted to RGB 3-channel and resized to 32×32.

### Data loading
- Uses `ImglistDataset` with `.txt` imglist files
- Batch size: 200 (from configs)
- No data augmentation for test/OOD

---

## Checkpoint / Model Loading

### Single run (from `cifar10_test_ood_ebo.sh`)
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

### Multi-run (from `scripts/eval_ood.py`)
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

The `--root` directory should contain subdirectories `s0/`, `s1/`, etc., each with `best.ckpt`.

### Network architecture
- `ResNet18_32x32` (from `openood/networks/`)
- 10 output classes for CIFAR-10
- Checkpoint path: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`

---

## AUROC & Aggregation Semantics

### Metrics computed (from `openood/evaluators/metrics.py`)
- **AUROC** (Area Under ROC Curve)
- **AUPR** (Area Under Precision-Recall curve)
- **FPR@95** (False Positive Rate at 95% True Positive Rate)
- **Detection Error**

### Aggregation (from `scripts/eval_ood.py`)
```python
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)
```

- Metrics are computed per-run, then **mean ± std** across runs
- Output format: `"XX.XX ± Y.YY"` for each metric
- Results saved as CSV if `--save-csv` is set

### Score caching
- Scores are saved to `{subfolder}/scores/ebo.pkl` (pickle)
- Postprocessor state saved to `{subfolder}/postprocessors/ebo.pkl`

---

## CPU / Dependency Risks

### CUDA dependency
- `scripts/eval_ood.py` uses `torch.cuda` implicitly via model `.to(device)`
- **No CPU fallback** in the current code
- To run on CPU: modify `scripts/eval_ood.py` to set `device = 'cpu'` and remove `.cuda()` calls

### Dependencies
- PyTorch (with CUDA recommended)
- torchvision ≥ 0.13 (for `InterpolationMode.BILINEAR`)
- NumPy, Pandas, pickle
- OpenOOD package (local import from `openood/`)

### Potential issues
1. **CUDA out of memory**: Batch size 200 may be large for some GPUs
2. **Missing imglist files**: Ensure `./data/benchmark_imglist/` exists
3. **Checkpoint path**: Must match exactly (including `s0/` subdirectory)
4. **torchvision version**: `InterpolationMode` requires torchvision ≥ 0.13

---

## Minimal Implementation Plan

### Step 1: Environment Setup
```bash
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -r requirements.txt
```

### Step 2: Data Preparation
- Download CIFAR-10, CIFAR-100, TinyImageNet, MNIST, SVHN, Texture, Places365
- Place in `./data/images_classic/`
- Ensure imglist files in `./data/benchmark_imglist/cifar10/`

### Step 3: Get Checkpoint
- Train CIFAR-10 ResNet18_32x32 baseline (100 epochs, lr=0.1)
- Or download pre-trained checkpoint to:
  `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`

### Step 4: Single Run Test
```bash
bash scripts/ood/ebo/cifar10_test_ood_ebo.sh
```

### Step 5: Multi-Run Evaluation
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Step 6: CPU Adaptation (if needed)
- Modify `scripts/eval_ood.py` line ~150: change `device = 'cuda'` to `device = 'cpu'`
- Remove `.cuda()` calls on model and data

### Step 7: Verify Output
- Check printed metrics table (AUROC, AUPR, FPR@95)
- Verify CSV saved to `{root}/ood/ebo.csv`

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Batch size | 200 |
| Image size | 32×32 |
| Normalization | CIFAR-10 mean/std |
| Number of runs | All subdirectories in `--root` |
| Checkpoint file | `best.ckpt` per run directory |
