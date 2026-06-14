## Handoff

**Entry point:** `tools/test.py`

**Config path:** `configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model definition)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset, batch size 16, test split at `data/cifar10/`)
- `configs/_base_/schedules/cifar10_bs128.py` (training schedule, not used for eval)
- `configs/_base_/default_runtime.py` (runtime defaults)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**What it does:**
- Loads the config, which sets `test_dataloader` to use `data/cifar10/` test split (10,000 images)
- Loads the checkpoint from `ckpt.pth`
- Runs evaluation with `Accuracy` metric (top-1)
- Prints `accuracy/top1:` with the percentage on the 10,000 test images
