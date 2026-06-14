## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 40-84, 113-146)
- **Shell example**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh` (lines 17-33)
- **Checkpoint layout**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Config template**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/config.yml` (lines 1-54)

### EBO & AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses `postprocessors/ebo.yml` with temperature=1. Energy score = `T * logsumexp(logits / T)`. Higher energy → more OOD-like.
- **AUROC**: Area Under the Receiver Operating Characteristic curve, reported as percentage (e.g., 94.70%). The `Evaluator` in `openood/evaluation_api` computes this automatically for Near-OOD splits.

### Data & Preprocessing
- **ID data**: CIFAR-10 (10 classes), 32×32 images, CIFAR-10 normalization, `ImglistDataset` with `./data/images_classic/` and `./data/benchmark_imglist/cifar10/`
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (automatically loaded by `get_ood_dataloader` based on config)
- **Preprocessor**: `base_preprocessor.yml` (standard normalization, no augmentations for test)

### Checkpoint Layout
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
├── s0/best.ckpt   (seed 0, 95.21% test accuracy)
├── s1/best.ckpt   (seed 1)
└── s2/best.ckpt   (seed 2)
```
Each subfolder contains `config.yml` and `best.ckpt`. The `eval_ood.py` script expects this `s*` subfolder structure.

### Aggregation
- **Per-seed**: AUROC computed individually for s0, s1, s2
- **Final**: Mean ± std across the 3 seeds (standard OpenOOD reporting)
- The `Evaluator` in `eval_ood.py` handles this automatically when `--root` points to the parent directory containing `s0/`, `s1/`, `s2/`

### CPU/Dependency Risks
- **CPU-only**: Set `--num_workers 0` (or 1-2) to avoid multiprocessing issues. The shell example uses `--num_workers 8` which may fail on CPU-only.
- **Offline**: All data and checkpoints are local. No internet access needed.
- **PyTorch**: Must be installed with CPU-only support (no CUDA). The code uses `torch.load(..., map_location='cpu')` implicitly via `Evaluator`.
- **Memory**: Each forward pass on 32×32 images with ResNet-18 is lightweight (~200MB). Batch size 200 is safe on CPU.

### Reproduction Command
```bash
cd /path/to/openood
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --num_workers 0 \
    --save-score --save-csv
```
This will:
1. Load checkpoints from `s0/`, `s1/`, `s2/`
2. Evaluate EBO with temperature=1 (default in postprocessor config)
3. Compute AUROC for Near-OOD (CIFAR-100, TinyImageNet)
4. Print per-seed and mean±std results as percentages

### Expected Output Format
```
FSOOD: False
Postprocessor: ebo
...
Near-OOD AUROC: 94.70 ± 0.15
```
(Actual numbers will vary; this is illustrative of the reporting format.)
