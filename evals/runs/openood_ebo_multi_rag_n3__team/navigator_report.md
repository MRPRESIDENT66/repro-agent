## Handoff Report: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet Near-OOD datasets in CPU-only, offline environment.

### Repository Structure & Key Files

**Evaluation Scripts:**
- `scripts/eval_ood.py` - Unified evaluator (recommended approach)
- `scripts/ood/ebo/cifar10_test_ood_ebo.sh` - Original shell script with both methods

**Postprocessor:**
- `openood/postprocessors/ebo_postprocessor.py` - EBO implementation using `temperature * logsumexp(output / temperature, dim=1)`

**Evaluation API:**
- `openood/evaluation_api/evaluator.py` - `Evaluator` class with `compute_all_metrics`
- `openood/evaluators/ood_evaluator.py` - `OODEvaluator` with `_eval_ood` method

**Checkpoints Location:**
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt`

### EBO & AUROC Semantics

**EBO Score:** `temperature * logsumexp(logits / temperature, dim=1)` where temperature=1 (default). Higher scores indicate more ID-like samples.

**AUROC Calculation:** From `openood/evaluators/ood_evaluator.py` lines 137-143:
- OOD labels hard-coded as -1
- ID labels from ground truth
- `compute_all_metrics(conf, label, pred)` returns metrics tuple where `ood_metrics[1]` is AUROC
- Final output: `100 * val_auroc` (percentage)

### Data & Preprocessing

**ID Data (CIFAR-10):**
- `data/images_classic/` with imglist at `data/benchmark_imglist/cifar10/`
- Image size: 32x32, normalization: cifar10 (per-channel mean/std)
- Batch size: 200 for evaluation

**Near-OOD Datasets:**
- CIFAR-100 and TinyImageNet (automatically loaded by `get_ood_dataloader` or `data_setup`)
- Both are Near-OOD splits per OpenOOD benchmark

### Execution Plan

**Recommended approach** (from shell script comments and `eval_ood.py`):

```bash
# For each seed s0, s1, s2:
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This automatically:
1. Loads all checkpoints from the root directory (s0, s1, s2)
2. Uses EBO postprocessor with temperature=1
3. Evaluates on CIFAR-100 and TinyImageNet Near-OOD
4. Reports percentage AUROC

**Alternative** (single seed via `main.py`):
```bash
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 4 \
    --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --postprocessor.postprocessor_args.temperature 1
```

### Aggregation

The `eval_ood.py` script handles multi-seed aggregation automatically when pointing to the root directory containing s0, s1, s2 subdirectories. Results are averaged across seeds.

### CPU/Dependency Risks

1. **CPU-only:** Set `--num_workers 4` (or lower) to avoid multiprocessing issues. The code uses `torch.no_grad()` which is CPU-compatible.
2. **Offline:** All data and checkpoints are pre-downloaded. No external downloads needed.
3. **Dependencies:** Requires PyTorch, numpy, pickle (for score caching). All should be present in the fixed environment.

### Expected Output Format

The evaluator prints per-dataset AUROC (percentage) for Near-OOD:
- CIFAR-100: ~XX.XX%
- TinyImageNet: ~XX.XX%

Final aggregated results across s0, s1, s2 will be printed as mean ± std.
