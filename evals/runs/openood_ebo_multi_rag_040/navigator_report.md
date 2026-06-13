## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command

```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 24-28)

### EBO and AUROC Semantics

- **EBO (Energy-Based OOD Detection):** Uses the negative energy score `-E(x) = logsumexp(f(x)/T)` where `f(x)` are logits and `T=1` (default temperature). Higher energy → more OOD-like.
- **AUROC:** Computed via `sklearn.metrics.roc_curve` treating OOD as positive class. The `-conf` (negated energy) is passed as scores. ID samples have larger conf values than OOD samples, so negation is applied internally in `metrics.py` (line: `fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)`).
- **Output:** Percentage AUROC (0-100 scale) reported per Near-OOD dataset.

### Data and Preprocessing

- **ID Data:** CIFAR-10 (32x32, bilinear interpolation, CIFAR-10 normalization)
- **Near-OOD Datasets:** CIFAR-100 and TinyImageNet (both 32x32)
- **Data Root:** `./data/images_classic/`
- **Imglist Paths:** `./data/benchmark_imglist/cifar10/` for ID; OOD datasets use OpenOOD's internal setup via `data_setup()` in `evaluation_api/datasets.py`
- **Preprocessor:** `base_preprocessor.yml` (standard normalization, no augmentations during eval)
- **Batch Size:** 200 (default in `eval_ood.py`)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Seed 0 checkpoint
│   ├── config.yml         # Seed 0 config
│   └── log.txt            # Training log (best acc 94.70 at epoch 96)
├── s1/
│   ├── best.ckpt          # Seed 1 checkpoint
│   └── config.yml         # Seed 1 config
└── s2/
    ├── best.ckpt          # Seed 2 checkpoint
    └── config.yml         # Seed 2 config
```

**Source:** `scripts/eval_ood.py` lines 76-84 (iterates over `s*` subfolders)

### Aggregation

The `eval_ood.py` script automatically:
1. Iterates over `s0`, `s1`, `s2` subfolders
2. Loads each `best.ckpt` checkpoint
3. Runs EBO evaluation on both Near-OOD datasets
4. Computes per-seed metrics (AUROC, FPR, AUPR)
5. Averages metrics across seeds and prints final results

**Source:** `scripts/eval_ood.py` lines 82-84, 113-124

### CPU/Dependency Risks

- **CPU-only:** The `eval_ood.py` script uses `torch` but does not explicitly call `.cuda()`. However, the `BaseEvaluator` in `openood/evaluators/base_evaluator.py` (line 28) calls `data.cuda()`. The `Evaluator` class in `evaluation_api/evaluator.py` does NOT call `.cuda()` — it uses the model as-is. **Risk:** If the model was saved with CUDA parameters, loading on CPU may fail. **Mitigation:** Ensure `map_location='cpu'` is used when loading checkpoints (the `Evaluator` class handles this internally via `torch.load(ckpt_path, map_location='cpu')`).
- **Offline:** All data and checkpoints are local. No internet access needed.
- **Dependencies:** `torch`, `numpy`, `sklearn`, `tqdm`, `pyyaml` — all standard in OpenOOD environment.

### Expected Output Format

```
Processing nearood...
Performing inference on cifar100 dataset...
...
Performing inference on tinyimagenet dataset...
...
Results:
| Dataset | FPR@95 | AUROC | AUPR_IN | AUPR_OUT | ACC |
|---------|--------|-------|---------|----------|-----|
| cifar100| ...    | XX.XX | ...     | ...      | ... |
| tinyimagenet| ... | XX.XX | ...     | ...      | ... |
```

The AUROC values are reported as percentages (0-100 scale).
