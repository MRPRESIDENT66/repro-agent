## Grounded Handoff

### Exact Source Paths
- **Eval script**: `scripts/eval_ood.py` (lines 39-55 for args, lines 63-65 for model mapping)
- **EBO config**: `configs/postprocessors/ebo.yml` (temperature=1, APS_mode=True)
- **OOD dataset config**: `configs/datasets/cifar10/cifar10_ood.yml` (nearood: [cifar100, tin])
- **Checkpoint layout**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Shell reference**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 24-33 show both main.py and eval_ood.py approaches)

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses logit energy `E(x) = -T * log(sum(exp(logits/T)))` as OOD score. Lower energy = ID, higher energy = OOD. Temperature=1 from config.
- **AUROC**: Area Under ROC curve, reported as percentage (e.g., 94.70%). Higher is better for OOD detection.
- **Near-OOD**: CIFAR-100 (semantically similar classes) and TinyImageNet (downsampled ImageNet) as OOD datasets.

### Data and Preprocessing
- **ID data**: CIFAR-10 test set (10 classes, 32x32 images)
- **OOD data**: CIFAR-100 and TinyImageNet from `./data/images_classic/` with imglist paths in `cifar10_ood.yml`
- **Preprocessing**: `base_preprocessor.yml` (standard normalization, no augmentation for test)
- **Batch size**: 128 for OOD evaluation (from `cifar10_ood.yml`)

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt
├── s1/best.ckpt
└── s2/best.ckpt
```
Each checkpoint is a ResNet-18_32x32 trained with CrossEntropy for 100 epochs, achieving ~94.7% validation accuracy.

### Aggregation
- **Per-seed**: Run EBO evaluation separately for s0, s1, s2
- **Final**: Average AUROC across 3 seeds for each OOD dataset (CIFAR-100 and TinyImageNet)
- Use `scripts/eval_ood.py` with `--root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default --postprocessor ebo --save-score --save-csv` for automated multi-seed aggregation

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or 1 to avoid multiprocessing issues; remove GPU flags from shell scripts
- **Offline**: All data and checkpoints must be pre-downloaded; no internet access during evaluation
- **Dependencies**: PyTorch, torchvision, numpy, pickle (for score caching), OpenOOD package installed
- **Memory**: ResNet-18_32x32 is lightweight; batch_size=128 should fit CPU memory

### Reproduction Command
```bash
PYTHONPATH='.':$PYTHONPATH python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv \
    --num_workers 0
```
This will evaluate all 3 seeds (s0, s1, s2) and output per-OOD-dataset AUROC percentages averaged across seeds.
