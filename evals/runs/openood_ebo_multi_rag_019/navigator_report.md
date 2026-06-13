## Handoff Report: EBO Near-OOD AUROC Reproduction (CIFAR-10)

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet (Near-OOD). CPU-only, offline environment.

### Key Semantics

**EBO Score Definition** (from `openood/postprocessors/ebo_postprocessor.py`):
- Score = `temperature * logsumexp(logits / temperature, dim=1)`
- Default temperature = 1 (set in `scripts/ood/ebo/cifar10_test_ood_ebo.sh`)
- Higher score → more ID-like; lower score → more OOD-like

**AUROC Computation** (from `openood/evaluators/metrics.py`):
- OOD samples labeled as positive class (ood_indicator=1 for label==-1)
- Uses `sklearn.metrics.roc_curve(ood_indicator, -conf)` — negates confidence because ID samples have higher scores
- AUROC reported as percentage (multiplied by 100 in `ood_evaluator.py` line 142)

### Data & Preprocessing

**ID Data**: CIFAR-10 test set (10,000 images)
- Config: `configs/datasets/cifar10/cifar10_ood.yml`
- Image size: 32×32, normalization: CIFAR-10 mean/std
- Batch size: 200 (test), 128 (ood)

**Near-OOD Datasets** (from `cifar10_ood.yml`):
- **CIFAR-100**: `data/benchmark_imglist/cifar10/test_cifar100.txt`
- **TinyImageNet (tin)**: `data/benchmark_imglist/cifar10/test_tin.txt`
- Data directory: `./data/images_classic/`

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt
├── s1/best.ckpt
└── s2/best.ckpt
```

Each checkpoint is a standard PyTorch state_dict for ResNet-18_32x32 (10 classes).

### Recommended Evaluation Method

Use `scripts/eval_ood.py` (unified evaluator) for each seed:

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This automatically:
1. Loads all three seeds (s0, s1, s2) from the root directory
2. Uses `best.ckpt` from each subfolder
3. Applies EBO postprocessor with temperature=1
4. Evaluates on Near-OOD (CIFAR-100, TinyImageNet) and Far-OOD
5. Aggregates metrics across seeds (mean ± std)

### Aggregation

The `eval_ood.py` script handles multi-seed aggregation internally. Results are printed as mean ± std across seeds for each metric/dataset pair.

### CPU & Dependency Risks

1. **CPU-only**: Set `--num_workers 0` or low value (e.g., 2) to avoid multiprocessing issues. The `eval_ood.py` script uses `num_workers=4` by default.
2. **Offline**: All data and checkpoints must be present at expected paths. No internet access needed.
3. **Dependencies**: Requires `torch`, `numpy`, `sklearn`, `tqdm`, `pyyaml` — verify these are installed.
4. **Memory**: Each seed evaluation requires ~2-4GB RAM for ResNet-18 on CPU.

### Execution Command (CPU-safe)

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

### Expected Output Format

The script prints per-dataset metrics including:
- `FPR@95`
- `AUROC` (percentage)
- `AUPR_IN`
- `AUPR_OUT`
- `ACC`

For Near-OOD, you'll see results for `cifar100` and `tin` under the "nearood" section, with mean ± std across the three seeds.

### Verification

To confirm correct execution, check that:
1. Scores are saved to `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/scores/ebo.pkl` (and similarly for s1, s2)
2. CSV output is generated with all metrics
3. AUROC values match the official OpenOOD benchmark for EBO on CIFAR-10 Near-OOD
