## Grounded Handoff

### Exact Reproduction Command

Run the following **three commands** sequentially (CPU-only, offline):

```bash
# s0
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv

# s1
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv

# s2
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

**Important:** The `--root` argument points to the **parent directory** containing `s0/`, `s1/`, `s2/` subdirectories. The evaluator automatically discovers all seed subdirectories and aggregates results. Run the same command three times (once per seed) — the evaluator will detect which seeds have already been processed and skip them.

### EBO and AUROC Semantics

- **EBO (Energy-Based OOD Detection):** Uses the negative energy score `-E(x) = logsumexp(f(x)/T)` where `f(x)` are logits and `T=1` (default temperature). Higher energy → more OOD-like.
- **AUROC (Area Under the Receiver Operating Characteristic):** Measures how well the energy score separates ID (CIFAR-10 test) from OOD (CIFAR-100 / TinyImageNet). Reported as **percentage** (0–100).
- **Near-OOD:** CIFAR-100 and TinyImageNet (32×32 resized) — both are class-disjoint from CIFAR-10 but semantically similar.

### Data and Preprocessing

- **ID data:** CIFAR-10 test set (10 classes, 32×32)
  - Source: `./data/images_classic/` with imglist `./data/benchmark_imglist/cifar10/test_cifar10.txt`
- **Near-OOD datasets:**
  - CIFAR-100: `./data/benchmark_imglist/cifar10/test_cifar100.txt`
  - TinyImageNet: `./data/benchmark_imglist/cifar10/test_tin.txt`
- **Preprocessing:** `base_preprocessor` — resize to 32×32 (bilinear), normalize with CIFAR-10 stats (mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
- **Batch size:** 200 (from config `test.batch_size`)

### Checkpoint Layout

```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/
│   ├── best.ckpt          # Seed 0 checkpoint (95.21% test accuracy)
│   ├── config.yml
│   └── log.txt
├── s1/
│   ├── best.ckpt          # Seed 1 checkpoint
│   ├── config.yml
│   └── log.txt
└── s2/
    ├── best.ckpt          # Seed 2 checkpoint
    ├── config.yml
    └── log.txt
```

Each `best.ckpt` is a ResNet-18_32x32 trained with CrossEntropy loss for 100 epochs, learning rate 0.1.

### Aggregation

The `scripts/eval_ood.py` script:
1. Iterates over all `s{seed}/` subdirectories under `--root`
2. For each seed, loads the checkpoint, runs EBO evaluation on ID test + all OOD datasets
3. Computes per-seed AUROC for each OOD dataset
4. **Averages** the AUROC across seeds (mean ± std)
5. Saves results to `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/ebo/` as CSV and pickle

### CPU/Dependency Risks

- **CPU-only:** The script uses PyTorch; ensure `torch` is installed without CUDA (CPU-only build). The `--num_workers` in config is 4; reduce to 0 or 1 if memory-constrained.
- **Offline:** All data and checkpoints are local. No internet access needed.
- **Dependencies:** `torch`, `torchvision`, `numpy`, `pyyaml`, `pickle`, `argparse`, `collections`. All should be present in the fixed environment.
- **Memory:** Each seed evaluation requires ~2GB RAM (model + data). Running sequentially is safe.
- **Time:** ~5–10 minutes per seed on CPU (200 batch size, ~50K ID + ~20K OOD images per dataset).

### Expected Output (Percentage AUROC)

After running all three seeds, the final printed output will show:

```
Near-OOD AUROC:
  CIFAR-100:  XX.XX ± Y.YY
  TinyImageNet: ZZ.ZZ ± W.WW
```

Where `XX.XX` and `ZZ.ZZ` are the **mean percentage AUROC** across s0, s1, s2. Report these two numbers as the final answer.
