## Handoff

**Entry point:** `tools/test.py`

**Config path:** `configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model definition)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset, evaluator)
- `configs/_base_/schedules/cifar10_bs128.py` (training schedule)
- `configs/_base_/default_runtime.py` (runtime defaults)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**Key details:**
- `tools/test.py` accepts `config` and `checkpoint` as positional args
- The config's `test_dataloader` uses `data/cifar10/` with `split='test'`, batch_size=16
- The evaluator is `Accuracy` with `topk=(1,)`, printing `accuracy/top1:`
- Test set has 10,000 images (CIFAR-10 test split)
- CPU-only: no special flags needed (default launcher is `none`)
