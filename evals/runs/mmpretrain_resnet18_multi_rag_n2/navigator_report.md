## Handoff

**Entry point:** `tools/test.py`

**Config path:** `configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model definition)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset, batch size 16, data at `data/cifar10/`, test set 10000 images)
- `configs/_base_/schedules/cifar10_bs128.py` (training schedule, not used for eval)
- `configs/_base_/default_runtime.py` (runtime defaults)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**Metric printed:** `accuracy/top1:` (percentage on 10000 test images)

**Data location:** `data/cifar10/` (already provisioned)

**Checkpoint location:** `ckpt.pth` (already provisioned)
