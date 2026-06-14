## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 40-84, 113-124, 139-150)
- **Shell example**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 17-33)
- **Checkpoint configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`, `s1/config.yml`, `s2/config.yml`
- **Checkpoint files**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`, `s1/best.ckpt`, `s2/best.ckpt`
- **Data root**: `./data/images_classic/` with imglist paths in `./data/benchmark_imglist/cifar10/`

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses logit energy `E(x) = -T * log(sum(exp(logits/T)))` as OOD score. Temperature=1 (from shell script `--postprocessor.postprocessor_args.temperature 1`).
- **AUROC**: Area Under the Receiver Operating Characteristic curve, reported as percentage (e.g., 95.21% accuracy in log.txt). The `Evaluator` in `openood/evaluation_api` computes this automatically.
- **Near-OOD**: Uses CIFAR-100 and TinyImageNet as OOD datasets against CIFAR-10 ID.

### Data and Preprocessing
- **ID dataset**: CIFAR-10 (10 classes, 32x32 images)
- **OOD datasets**: CIFAR-100 and TinyImageNet (both 32x32, via `cifar10_ood.yml` config)
- **Preprocessor**: `base_preprocessor.yml` (standard CIFAR-10 normalization)
- **Data format**: ImglistDataset with `.txt` files listing image paths
- **Batch size**: 200 for test/ood evaluation

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # seed 0 checkpoint
│   ├── config.yml         # seed 0 config
│   └── log.txt            # training log (95.21% test accuracy)
├── s1/
│   ├── best.ckpt          # seed 1 checkpoint
│   └── config.yml         # seed 1 config
└── s2/
    ├── best.ckpt          # seed 2 checkpoint
    └── config.yml         # seed 2 config
```

### Aggregation
- **Per-seed**: Run `scripts/eval_ood.py` separately for each seed (s0, s1, s2)
- **Final**: Average AUROC across 3 seeds, report as percentage (e.g., `mean ± std`)

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` or low value (e.g., 2) to avoid multiprocessing issues. The script uses PyTorch which will run on CPU by default.
- **Offline**: All data and checkpoints are local. No network calls needed.
- **Dependencies**: Requires `torch`, `numpy`, `pyyaml`, `glob`, `os` (all standard in OpenOOD environment)

### Reproduction Command
```bash
cd /path/to/openood
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv \
    --num_workers 0
```
This will iterate over s0, s1, s2 subfolders, compute EBO AUROC for each, and print aggregated results. The `--save-score` flag enables score saving for verification.
