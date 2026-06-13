## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command
```bash
cd /path/to/openood  # repository root

# Run for each seed (s0, s1, s2) - adjust checkpoint path per seed
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Checkpoint Layout
- **Path**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Architecture**: ResNet18_32x32 (10 classes, CIFAR-10 variant)
- **Training**: CrossEntropy, 100 epochs, lr=0.1, seed={0,1,2}
- **Validation accuracy**: ~94.70% (s0), similar for s1/s2

### Data & Preprocessing
- **ID data**: CIFAR-10 (test split, 10k images)
- **Near-OOD datasets** (from `configs/datasets/cifar10/cifar10_ood.yml`):
  - **CIFAR-100**: `./data/benchmark_imglist/cifar10/test_cifar100.txt`
  - **TinyImageNet**: `./data/benchmark_imglist/cifar10/test_tin.txt`
- **Image size**: 32×32, bilinear interpolation
- **Normalization**: CIFAR-10 specific (`normalization_type: cifar10`)
- **Batch size**: 200 (eval)

### EBO & AUROC Semantics
- **EBO (Energy-Based OOD)**: Score = `-temperature * logsumexp(logits / temperature)`, temperature=1 (default)
- **AUROC**: Area Under ROC curve; higher = better separation between ID (CIFAR-10 test) and OOD samples
- **Near-OOD**: Only CIFAR-100 and TinyImageNet (not far-OOD like MNIST/SVHN)

### Aggregation
- **Per-seed**: Run `eval_ood.py` once per seed (s0, s1, s2)
- **Final metric**: Mean ± std across 3 seeds for each Near-OOD dataset
- **Output**: `--save-csv` generates CSV with per-dataset AUROC; `--save-score` caches logits

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or 1 to avoid multiprocessing issues; the script uses PyTorch which will fall back to CPU automatically
- **Offline**: All data and checkpoints must be pre-downloaded; no internet access needed
- **Dependencies**: PyTorch, torchvision, numpy, scikit-learn, PyYAML, pickle (standard OpenOOD stack)
- **Memory**: ~2GB RAM sufficient for batch_size=200 on CPU

### Expected Output Format
```
# Near-OOD AUROC (%):
# CIFAR-100:  X.XX ± Y.YY
# TinyImageNet:  A.AA ± B.BB
```

### Verification Steps
1. Confirm checkpoint exists at `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
2. Confirm OOD data lists exist at paths in `cifar10_ood.yml`
3. Run `eval_ood.py` for each seed
4. Parse CSV output for Near-OOD AUROC values
5. Report mean ± std across seeds as percentage
