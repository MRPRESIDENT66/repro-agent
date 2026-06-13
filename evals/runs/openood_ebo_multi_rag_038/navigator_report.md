## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Source Paths

**Primary evaluation script:** `scripts/eval_ood.py` (lines 146-157, 167-205)
**EBO postprocessor:** `openood/postprocessors/ebo_postprocessor.py` (full file)
**OOD config:** `configs/datasets/cifar10/cifar10_ood.yml` (full file)
**Evaluation API:** `openood/evaluation_api/evaluator.py` (lines 251-300)
**Metrics:** `openood/evaluators/metrics.py` (compute_all_metrics)

### EBO and AUROC Semantics

**EBO (Energy-Based OOD) score** is computed as:
```
conf = temperature * logsumexp(logits / temperature, dim=1)
```
Higher energy → more OOD-like. The postprocessor returns `(pred, conf)` where `conf` is the energy score. AUROC is computed by treating higher energy as the OOD indicator.

**Default temperature = 1** (from `scripts/ood/ebo/cifar10_test_ood_ebo.sh` line 24).

### Data and Preprocessing

**ID data:** CIFAR-10 test set (10 classes)
**Near-OOD datasets** (from `cifar10_ood.yml`):
- **CIFAR-100:** `./data/images_classic/` with imglist `./data/benchmark_imglist/cifar10/test_cifar100.txt`
- **TinyImageNet (tin):** `./data/images_classic/` with imglist `./data/benchmark_imglist/cifar10/test_tin.txt`

**Preprocessing:** Default (via `preprocessor=None` in `Evaluator.__init__`), which follows OpenOOD conventions for CIFAR-10 (normalization with mean/std).

### Checkpoint Layout

Checkpoints are at:
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt
```

Each is a standard PyTorch state_dict for `ResNet18_32x32` (loaded via `torch.load(..., map_location='cpu')`).

### Aggregation

The `scripts/eval_ood.py` script processes all subfolders (s0, s1, s2) and aggregates results. From lines 146-157, it loops over subfolders, loads each checkpoint, creates an `Evaluator`, and calls `evaluator.eval_ood()`. The `Evaluator.eval_ood()` method (evaluator.py lines 251-300) computes per-dataset metrics including AUROC for each OOD dataset separately.

**Output format:** The script prints metrics per OOD dataset (CIFAR-100 and TinyImageNet separately) as percentages. The final reported AUROC for each dataset is the average across the three seeds (s0, s1, s2).

### CPU/Dependency Risks

1. **CPU-only constraint:** The `base_postprocessor.py` `inference()` method calls `.cuda()` on data (line 40). The `scripts/eval_ood.py` also calls `net.cuda()` (line 148). **Must modify** these to use CPU (remove `.cuda()` calls or add `map_location='cpu'` and remove CUDA transfers).

2. **Offline environment:** All data and checkpoints are already present at the paths specified in configs. No downloads needed.

3. **Dependencies:** Requires PyTorch, numpy, tqdm, pickle, yaml (standard OpenOOD stack). All should be available in the fixed environment.

### Execution Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Before running:** Patch `openood/postprocessors/base_postprocessor.py` line 40 and `scripts/eval_ood.py` line 148 to remove `.cuda()` calls (replace with CPU-compatible operations). The EBO postprocessor itself (`ebo_postprocessor.py`) has no CUDA calls.

### Expected Output

The script will print AUROC percentages for each OOD dataset (CIFAR-100, TinyImageNet) averaged over s0/s1/s2. Report as percentage (e.g., "AUROC: 85.23%").
