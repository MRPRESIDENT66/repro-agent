## Grounded Handoff: EBO Near-OOD AUROC for CIFAR-10

### Exact Reproduction Command

```bash
# For each seed (s0, s1, s2), run:
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Source:** `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 30-35)

### EBO and AUROC Semantics

- **EBO (Energy-Based OOD Detection):** Uses the negative energy score `-E(x) = logsumexp(f(x)/T)` where `f(x)` are logits and `T=1` (default temperature). Higher energy → more OOD-like.
- **AUROC (Area Under the Receiver Operating Characteristic):** Measures separability between ID (CIFAR-10 test) and OOD (CIFAR-100, TinyImageNet) energy scores. Reported as percentage (0-100).

### Data and Preprocessing

- **ID Data:** CIFAR-10 test split (`./data/benchmark_imglist/cifar10/test_cifar10.txt`)
- **Near-OOD Datasets:** CIFAR-100 and TinyImageNet (automatically loaded by `Evaluator` via `DATA_INFO` in `openood/evaluation_api/datasets.py`)
- **Preprocessing:** `base_preprocessor` with CIFAR-10 normalization (mean/std), image size 32×32, bilinear interpolation
- **Batch size:** 200 (default in `eval_ood.py`)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Seed 0 checkpoint
│   ├── config.yml         # Training config (seed=0)
│   └── log.txt            # Training log (best acc: 94.70)
├── s1/
│   ├── best.ckpt          # Seed 1 checkpoint
│   └── config.yml         # Training config (seed=1)
└── s2/
    ├── best.ckpt          # Seed 2 checkpoint
    └── config.yml         # Training config (seed=2)
```

**Source:** `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` (line 47-48)

### Aggregation

The `eval_ood.py` script (lines 82-84) iterates over `s*` subfolders, computes metrics per seed, and aggregates. The final reported AUROC is the **mean across s0, s1, s2** (standard deviation also reported in output).

### CPU/Dependency Risks

- **CPU-only:** The `Evaluator` class uses `torch.no_grad()` during inference. No GPU required, but expect ~5-10 minutes per seed on CPU.
- **Offline:** All data and checkpoints are local. No network calls needed.
- **Dependencies:** Requires `torch`, `numpy`, `tqdm`, `pyyaml`. All present in the fixed environment.
- **num_workers:** Default is 4 in `Evaluator.__init__`. For CPU-only, set `--num_workers 0` or 1 to avoid multiprocessing issues if needed.

### Expected Output Format

```
Processing nearood...
Performing inference on cifar100 dataset...
Performing inference on tinyimagenet dataset...
...
Results:
+----------------+--------+-------+--------+-------+
| Dataset        | FPR@95 | AUROC | AUPR_IN| AUPR_OUT|
+----------------+--------+-------+--------+-------+
| cifar100       |  X.XX  | XX.XX | XX.XX  | XX.XX  |
| tinyimagenet   |  X.XX  | XX.XX | XX.XX  | XX.XX  |
+----------------+--------+-------+--------+-------+
```

The Near-OOD AUROC for each dataset is reported as a percentage. The final handoff should report the mean AUROC across s0/s1/s2 for both CIFAR-100 and TinyImageNet.
