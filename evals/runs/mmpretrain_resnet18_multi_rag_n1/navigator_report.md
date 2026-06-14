## Handoff

**Entry point:** `tools/test.py`

**Config path:** `configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset)
- `configs/_base_/schedules/cifar10_bs128.py` (schedule)
- `configs/_base_/default_runtime.py` (runtime)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**Key details:**
- `tools/test.py` accepts `config` and `checkpoint` as positional args
- The config sets `test_dataloader` with `batch_size=16`, `data_root='data/cifar10/'`, `split='test'`
- The evaluator is `dict(type='Accuracy', topk=(1,))`
- Output metric line: `accuracy/top1: X.XX` (percentage on 10000 test images)
- CPU-only: default `--launcher none`; no GPU flags needed
