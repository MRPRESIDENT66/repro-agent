## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Source Paths

**Primary evaluation script:** `scripts/eval_ood.py` (unified evaluator, recommended)
**Alternative pipeline:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (single-run)
**EBO postprocessor:** `openood/postprocessors/ebo_postprocessor.py`
**OOD evaluator:** `openood/evaluators/ood_evaluator.py`
**Evaluation API:** `openood/evaluation_api/evaluator.py`
**OOD config:** `configs/datasets/cifar10/cifar10_ood.yml`
**Checkpoints root:** `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`

### EBO and AUROC Semantics

**EBO (Energy-Based OOD) score** = `temperature * logsumexp(logits / temperature, dim=1)` where `temperature=1` (default). Higher energy → more OOD-like. The `EBOPostprocessor.postprocess()` returns `(pred, conf)` where `conf` is this energy score (misleadingly named "conf").

**AUROC** is computed by `openood/evaluators/metrics.py` via `compute_all_metrics()`. The metric treats the energy score as the OOD detection score (higher = more OOD). AUROC is reported as a percentage (0-100).

### Data and Preprocessing

- **ID data:** CIFAR-10 test set (10 classes, 32×32)
- **Near-OOD datasets:** CIFAR-100 and TinyImageNet (tin)
- **Data location:** `./data/images_classic/`
- **Image list paths:**
  - ID test: `./data/benchmark_imglist/cifar10/test_cifar10.txt`
  - CIFAR-100 OOD: `./data/benchmark_imglist/cifar10/test_cifar100.txt`
  - TinyImageNet OOD: `./data/benchmark_imglist/cifar10/test_tin.txt`
- **Preprocessing:** `base_preprocessor.yml` → normalization_type: `cifar10`, image_size: 32, interpolation: bilinear
- **Batch size:** 200 (default in evaluator)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt   # seed 0, 94.70% val accuracy
├── s1/best.ckpt   # seed 1
└── s2/best.ckpt   # seed 2
```

Each checkpoint is a PyTorch state_dict for `ResNet18_32x32` (10-class output).

### Aggregation

The `scripts/eval_ood.py` script with `--root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default` automatically:
1. Iterates over `s0`, `s1`, `s2` subfolders
2. Loads each `best.ckpt`
3. Runs EBO evaluation per seed
4. **Averages metrics across seeds** (mean AUROC reported)

### CPU/Dependency Risks

1. **GPU hardcoding:** `base_postprocessor.py` line `data = batch['data'].cuda()` and `eval_ood.py` line `net.cuda()` will fail on CPU. Must modify to `data = batch['data']` and remove `.cuda()` calls.
2. **Offline:** All data and checkpoints are local; no downloads needed.
3. **Dependencies:** PyTorch, numpy, pickle, tqdm, PyYAML (all standard in OpenOOD env).

### Execution Plan

**Option A (Recommended - unified evaluator):**
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
After patching `cuda()` calls to CPU-compatible versions.

**Option B (Per-seed pipeline):**
```bash
for seed in s0 s1 s2; do
    python main.py \
        --config configs/datasets/cifar10/cifar10.yml \
        configs/datasets/cifar10/cifar10_ood.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/test/test_ood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/ebo.yml \
        --num_workers 4 \
        --network.checkpoint "results/cifar10_resnet18_32x32_base_e100_lr0.1_default/${seed}/best.ckpt" \
        --mark ${seed}
done
```
Then manually average the per-seed AUROC values from log files.

### Expected Output

The evaluator prints per-dataset AUROC for each near-OOD dataset (CIFAR-100, TinyImageNet) as percentages. The final reported values are the mean across 3 seeds.
