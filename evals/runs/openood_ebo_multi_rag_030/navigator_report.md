## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 40-84, 113-124, 139-150)
- **Shell example**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 17-34)
- **Checkpoint configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`, `s1/config.yml`, `s2/config.yml`
- **Pipeline**: `openood/pipelines/test_ood_pipeline.py` (full file)
- **Evaluation API**: `openood/evaluation_api` (imported in `eval_ood.py` line 15)

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses `configs/postprocessors/ebo.yml` with `--postprocessor.postprocessor_args.temperature 1`. Energy score = `T * logsumexp(logits / T)` where T=1.
- **AUROC**: Computed by `Evaluator` from `openood.evaluation_api`. The `eval_ood.py` script aggregates metrics across s0/s1/s2 subfolders and reports Near-OOD AUROC as a percentage (0-100 scale). The `--save-csv` flag outputs per-dataset results.

### Data and Preprocessing
- **ID data**: CIFAR-10 (10 classes, 32x32, `./data/images_classic/`, `./data/benchmark_imglist/cifar10/`)
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (automatically loaded by `get_ood_dataloader` based on `cifar10_ood.yml` config)
- **Preprocessor**: `base_preprocessor.yml` (CIFAR-10 normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
- **Batch size**: 200 (default in `eval_ood.py` line 47)

### Checkpoint Layout
- **Root**: `./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/`
- **Subfolders**: `s0/`, `s1/`, `s2/` (each contains `best.ckpt` and `config.yml`)
- **Network**: `ResNet18_32x32` (10 classes, no pretrained)
- **Checkpoint path per seed**: `{root}/s{seed}/best.ckpt`

### Aggregation
- `eval_ood.py` iterates over `s*` subfolders (line 84), evaluates each seed independently, then aggregates metrics (mean/std across seeds). Near-OOD AUROC is reported as the average across CIFAR-100 and TinyImageNet.

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` (or low value like 2) to avoid multiprocessing issues. The `eval_ood.py` script uses `--num_workers 8` by default in shell examples; override to 0.
- **Offline**: All data and checkpoints are local. No internet access needed.
- **Dependencies**: PyTorch, torchvision, numpy, PyYAML, glob, argparse (all standard in OpenOOD environment).

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

Expected output: Near-OOD AUROC ~94-95% (percentage scale) averaged across CIFAR-100 and TinyImageNet, with per-seed results printed and CSV saved.
