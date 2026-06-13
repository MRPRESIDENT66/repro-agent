To reproduce the official EBO Near-OOD AUROC for CIFAR-10 using the s0, s1, and s2 CrossEntropy ResNet-18 checkpoints:

**Key Components**
- **EBO Semantics**: Energy-Based Outlier detection uses temperature-scaled logsumexp of logits as confidence (openood/postprocessors/ebo_postprocessor.py). Default temperature=1.
- **AUROC Calculation**: Computed by OpenOOD's Evaluator between CIFAR-10 ID test set and Near-OOD datasets (CIFAR-100 and TinyImageNet) as defined in configs/datasets/cifar10/cifar10_ood.yml.
- **Data & Preprocessing**: 
  - ID data: ./data/benchmark_imglist/cifar10/test_cifar10.txt
  - Near-OOD: 
    - CIFAR-100: ./data/benchmark_imglist/cifar10/test_cifar100.txt
    - TinyImageNet: ./data/benchmark_imglist/cifar10/test_tin.txt
  - Base preprocessing (configs/preprocessors/base_preprocessor.yml) with CIFAR-10 normalization.
- **Checkpoints**: Located at:
  - results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
  - results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt
  - results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt
- **Aggregation**: Run all three seeds (s0,s1,s2) and average AUROC scores.

**Execution Command**
Use the unified evaluator script (recommended in cifar10_test_ood_ebo.sh):
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-csv
```

**Critical Notes**
1. **CPU Execution**: Ensure `num_gpus: 0` in config files or override via CLI (`--num_gpus 0`). The evaluator defaults to CPU when CUDA unavailable.
2. **Dependency Risks**: Requires exact OpenOOD environment (PyTorch, numpy, pandas). Offline mode assumes all data/checkpoints pre-downloaded.
3. **Output**: Results saved as CSV in checkpoint directories. Extract Near-OOD AUROC for cifar100/tin columns, convert to percentage (×100), then average across s0-s2.

Report final averaged percentage AUROC for both CIFAR-100 and TinyImageNet Near-OOD datasets.
