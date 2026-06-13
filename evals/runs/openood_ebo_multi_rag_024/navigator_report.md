## Grounded Handoff

### Exact Source Paths
- **Evaluation script**: `scripts/eval_ood.py` (lines 39-55, 63-68, 76-84)
- **EBO postprocessor**: `openood/postprocessors/ebo_postprocessor.py` (lines 1-30)
- **OOD evaluator**: `openood/evaluators/ood_evaluator.py` (lines 83-143)
- **Checkpoints**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`
- **Configs**: `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/config.yml`
- **Shell reference**: `scripts/ood/ebo/cifar10_test_ood_ebo.sh`

### EBO and AUROC Semantics
- **EBO (Energy-Based OOD)**: Uses `temperature * logsumexp(logits / temperature)` as the OOD score. Higher energy → more OOD. Temperature=1 is the default.
- **AUROC**: Computed by `compute_all_metrics` in `openood/evaluators/metrics.py`. The evaluator returns `100 * val_auroc` (percentage). Near-OOD AUROC is reported per dataset (CIFAR-100, TinyImageNet) then averaged across the two.

### Data and Preprocessing
- **ID data**: CIFAR-10 (32×32, bilinear interpolation, CIFAR-10 normalization)
- **Near-OOD datasets**: CIFAR-100 and TinyImageNet (resized to 32×32)
- **Data root**: `./data/images_classic/`
- **Imglist paths**: `./data/benchmark_imglist/cifar10/` for ID; OOD datasets use their own imglist files
- **Preprocessor**: `base_preprocessor` (standard normalization, no augmentations at test time)
- **Batch size**: 200 (default in eval_ood.py)

### Checkpoint Layout
- Three seeds: `s0`, `s1`, `s2` under `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/`
- Each contains `best.ckpt` (ResNet18_32x32, 10 classes, CrossEntropy trained)
- The `scripts/eval_ood.py` script expects this `s*` subfolder structure and iterates over them

### Aggregation
- **Per-seed**: AUROC computed per Near-OOD dataset (CIFAR-100, TinyImageNet)
- **Final**: Average across the two Near-OOD datasets for each seed, then average across s0, s1, s2 (standard OpenOOD practice)
- The `scripts/eval_ood.py` script handles this automatically when pointed at the root directory

### CPU/Dependency Risks
- **CPU-only**: The script uses `--num_workers 4` (default). Reduce to 0 or 1 if memory-constrained. Expect ~30-60 min per seed on CPU.
- **Offline**: All data and checkpoints are local. No network calls needed.
- **Dependencies**: PyTorch, torchvision, numpy, PyYAML, pickle (for score caching). All assumed present in the fixed environment.

### Reproduction Command
```bash
cd /path/to/openood
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
This will output per-seed, per-dataset AUROC values and the final averaged percentage. The `--save-csv` flag writes results to CSV files in each seed's subfolder.
