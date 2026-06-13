## Handoff Report: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet near-OOD datasets. CPU-only, offline environment.

### Repository Structure & Key Files

**Evaluation Scripts:**
- `scripts/eval_ood.py` - Unified evaluator (recommended approach)
- `scripts/ood/ebo/cifar10_test_ood_ebo.sh` - Original shell script with both methods

**Postprocessor:**
- `openood/postprocessors/ebo_postprocessor.py` - EBO implementation using `temperature * logsumexp(output / temperature, dim=1)`

**Evaluation API:**
- `openood/evaluation_api/evaluator.py` - `Evaluator` class
- `openood/evaluators/ood_evaluator.py` - `OODEvaluator` with `_eval_ood` method

**Checkpoints:**
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt`

### EBO & AUROC Semantics

**EBO Score:** `temperature * logsumexp(logits / temperature, dim=1)` where temperature=1 (default). Higher scores indicate in-distribution.

**AUROC:** Computed via `compute_all_metrics(conf, label, pred)` in `openood/evaluators/metrics.py`. OOD labels are hard-coded as -1. AUROC is multiplied by 100 for percentage output.

### Data & Preprocessing

**ID Data (CIFAR-10):**
- `data/images_classic/` with imglist at `data/benchmark_imglist/cifar10/`
- Image size: 32x32, normalization: cifar10 type
- Batch size: 200 for test

**Near-OOD Datasets:**
- CIFAR-100 and TinyImageNet (loaded automatically by `get_ood_dataloader` or `data_setup`)

**Preprocessor:** `base_preprocessor.yml` - standard normalization

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt
│   ├── config.yml
│   └── log.txt
├── s1/
│   ├── best.ckpt
│   └── config.yml
└── s2/
    ├── best.ckpt
    └── config.yml
```

### Aggregation Method
Run evaluation separately for each seed (s0, s1, s2), then average the AUROC values across seeds for each near-OOD dataset.

### Execution Command (per seed)
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Note:** This evaluates all checkpoints under the root directory. For individual seed evaluation, point `--root` to the specific seed directory (e.g., `./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0`).

### CPU/Dependency Risks
1. **CPU-only:** Expect significantly slower inference (CIFAR-10 test set: 10K images, near-OOD: ~20K images total). Consider reducing `--num_workers` to 0 or 1.
2. **Offline:** All data and checkpoints must be pre-downloaded. Verify `data/images_classic/` and `data/benchmark_imglist/` exist.
3. **Memory:** Each checkpoint ~44MB. Ensure sufficient RAM for model + data loading.
4. **Dependencies:** Requires PyTorch, torchvision, numpy, pickle, tqdm, PyYAML.

### Expected Output Format
The evaluator prints per-dataset metrics including AUROC (as percentage). The `--save-csv` flag saves results to CSV files. Aggregate by averaging AUROC across s0, s1, s2 for each near-OOD dataset.
