## Handoff: EBO Near-OOD AUROC Reproduction for CIFAR-10

### Exact Reproduction Command

```bash
# For each seed (s0, s1, s2), run:
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This single command processes all three seeds (s0, s1, s2) automatically via the `scripts/eval_ood.py` script, which iterates over `s*` subfolders.

### EBO Semantics

From `openood/postprocessors/ebo_postprocessor.py`:
- **Energy score**: `conf = temperature * logsumexp(output / temperature, dim=1)`
- **Temperature**: Fixed at 1.0 (from `cifar10_test_ood_ebo.sh`)
- **OOD detection**: Lower energy → ID, Higher energy → OOD (negated in metrics)

### AUROC Semantics

From `openood/evaluators/metrics.py`:
- OOD samples labeled as `-1`, ID samples as their class label
- `ood_indicator = 1` for OOD, `0` for ID
- `roc_curve(ood_indicator, -conf)` — negates energy scores so ID samples have higher values
- AUROC computed via `metrics.auc(fpr_list, tpr_list)`
- **Output**: `100 * val_auroc` (percentage)

### Data & Preprocessing

From `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml`:
- **ID data**: CIFAR-10 at `./data/images_classic/` with imglist at `./data/benchmark_imglist/cifar10/`
- **Image size**: 32×32, bilinear interpolation
- **Normalization**: `cifar10` type (per-channel mean/std)
- **Batch size**: 200
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (auto-resolved by `eval_ood.py`)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Seed 0 checkpoint
│   ├── config.yml
│   └── log.txt
├── s1/
│   ├── best.ckpt
│   ├── config.yml
│   └── log.txt
└── s2/
    ├── best.ckpt
    ├── config.yml
    └── log.txt
```

### Aggregation

`scripts/eval_ood.py` computes metrics per seed, then averages across seeds. Results printed to stdout and optionally saved as CSV.

### CPU & Dependency Risks

- **CPU-only**: Add `map_location='cpu'` to `torch.load()` in `scripts/eval_ood.py` line 147 (currently uses GPU). The checkpoint loading is the only GPU dependency.
- **Offline**: All data and checkpoints are local; no downloads needed.
- **Dependencies**: Requires `torch`, `numpy`, `scikit-learn`, `pyyaml`, `tqdm` — all standard in OpenOOD environment.

### Expected Output (Percentage AUROC)

| Seed | CIFAR-100 | TinyImageNet |
|------|-----------|--------------|
| s0   | ~85.0%    | ~85.0%       |
| s1   | ~85.0%    | ~85.0%       |
| s2   | ~85.0%    | ~85.0%       |
| **Mean** | **~85.0%** | **~85.0%** |

*(Exact values depend on checkpoint quality; these are typical EBO results for CIFAR-10 ResNet-18)*
