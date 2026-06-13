## Grounded Handoff

### Exact Source Paths

| Component | Path |
|-----------|------|
| EBO evaluation script | `scripts/ood/ebo/cifar10_test_ood_ebo.sh` |
| Unified evaluator | `scripts/eval_ood.py` |
| EBO postprocessor | `openood/postprocessors/ebo_postprocessor.py` |
| Test OOD pipeline | `openood/pipelines/test_ood_pipeline.py` |
| Checkpoints (s0/s1/s2) | `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt` |
| Config (s0 example) | `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` |

### EBO & AUROC Semantics

- **EBO (Energy-Based OOD)**: Uses `temperature * logsumexp(logits / temperature)` as the OOD score. Higher energy → more OOD. Default temperature = 1.0 (from `ebo_postprocessor.py` lines 14, 24-26).
- **AUROC**: Area Under the Receiver Operating Characteristic curve. Reported as percentage (0-100). The evaluator computes this automatically via `openood/evaluation_api/Evaluator`.

### Data & Preprocessing

- **ID data**: CIFAR-10 (10 classes, 32×32 images)
  - Train: `./data/images_classic/` + `./data/benchmark_imglist/cifar10/train_cifar10.txt`
  - Test: `./data/images_classic/` + `./data/benchmark_imglist/cifar10/test_cifar10.txt`
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (configured via `cifar10_ood.yml`)
- **Preprocessing**: `base_preprocessor.yml` → bilinear interpolation, normalization type `cifar10`, image size 32×32
- **Batch size**: 200 for test/val (from config)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt   (seed 0, best validation accuracy 94.70%)
├── s1/best.ckpt   (seed 1)
└── s2/best.ckpt   (seed 2)
```

Each checkpoint is a ResNet-18_32x32 trained with CrossEntropy loss for 100 epochs, lr=0.1.

### Aggregation

The unified evaluator (`scripts/eval_ood.py`) automatically:
1. Iterates over `s0`, `s1`, `s2` subfolders
2. Computes metrics per seed
3. Averages across seeds for final reported numbers

### CPU & Dependency Risks

- **CPU-only**: The `base_postprocessor.py` line 27 calls `.cuda()` which will fail. Must modify to use `.to('cpu')` or set `--num_gpus 0`. The `main.py` uses `launch()` with `config.num_gpus` – set to 0.
- **Offline**: All data and checkpoints are local. No downloads needed.
- **Dependencies**: PyTorch, torchvision, numpy, PyYAML, tqdm, scikit-learn (for AUROC). All assumed present.

### Reproduction Command (CPU-safe)

```bash
# Option A: Using main.py (modify for CPU)
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 4 \
    --num_gpus 0 \
    --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --postprocessor.postprocessor_args.temperature 1

# Option B: Using unified evaluator (recommended, handles all seeds)
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Critical**: For CPU, either:
- Patch `base_postprocessor.py` to replace `.cuda()` with `.cpu()`, OR
- Set environment variable `CUDA_VISIBLE_DEVICES=""` before running

### Expected Output

The evaluator prints per-seed and averaged AUROC (percentage) for each OOD dataset. The official reported values for EBO on CIFAR-10 with ResNet-18 are approximately:
- **CIFAR-100 Near-OOD**: ~80-82% AUROC
- **TinyImageNet Near-OOD**: ~85-87% AUROC

(Exact numbers depend on seed aggregation; the unified evaluator will compute and print these.)
