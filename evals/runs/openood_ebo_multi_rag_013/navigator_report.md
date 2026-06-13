## Handoff Report: EBO Near-OOD AUROC for CIFAR-10

### Task Summary
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using s0, s1, s2 CrossEntropy ResNet-18 checkpoints on CIFAR-100 and TinyImageNet. CPU-only, offline environment.

### Key Semantics

**EBO Score Definition** (`openood/postprocessors/ebo_postprocessor.py`):
```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```
- Default temperature = 1 (from `scripts/ood/ebo/cifar10_test_ood_ebo.sh`: `--postprocessor.postprocessor_args.temperature 1`)
- Higher score = more in-distribution (ID)

**AUROC Computation** (`openood/evaluators/ad_evaluator.py`):
- ID labels = 1, OOD labels = -1 (hard-coded)
- `roc_curve(ind_indicator, conf)` where `ind_indicator = (label != -1).astype(float)`
- Reports percentage: `100.0 * auroc`

### Data & Preprocessing

**OOD Datasets** (from `configs/datasets/cifar10/cifar10_ood.yml`):
- **Near-OOD**: `cifar100` and `tin` (TinyImageNet)
- Data root: `./data/images_classic/`
- Image lists: `./data/benchmark_imglist/cifar10/test_cifar100.txt` and `test_tin.txt`
- Batch size: 128, no shuffle
- Preprocessor: `base_preprocessor.yml` (standard normalization)

### Checkpoint Layout

Checkpoints at: `./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt`

### Execution Strategy

**Recommended**: Use `scripts/eval_ood.py` (unified evaluator):
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```
This processes all seeds (s0, s1, s2) automatically and aggregates results.

**Alternative**: Single-run per seed via `main.py`:
```bash
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 1 \
    --postprocessor.postprocessor_args.temperature 1
```

### Aggregation
- `eval_ood.py` averages AUROC across seeds for each OOD dataset
- Reports per-dataset and overall Near-OOD AUROC

### CPU & Dependency Risks
- **CPU-only**: Set `--num_workers 0` or low value (e.g., 2) to avoid multiprocessing issues
- **Offline**: All configs, data, and checkpoints are local; no downloads needed
- **Dependencies**: PyTorch, sklearn, numpy, PyYAML (standard OpenOOD stack)

### Expected Output Format
```
Complete Evaluation:
==============================
AUC Image: XX.XX 
==============================
```
Where XX.XX is the percentage AUROC (e.g., 85.23 for 0.8523).
