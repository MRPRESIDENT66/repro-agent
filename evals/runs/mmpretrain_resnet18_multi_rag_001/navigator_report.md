## Handoff: ResNet-18 CIFAR-10 Evaluation

### Entry Point
- **Path:** `mmpretrain/tools/test.py`
- **Usage:** `python tools/test.py <config> <checkpoint> [--work-dir <dir>]`
- **Positional args:** config path, checkpoint path
- **Output metric:** `accuracy/top1` (printed by mmengine Runner)

### Config
- **Main config:** `mmpretrain/configs/resnet/resnet18_8xb16_cifar10.py`
- **Inherited base configs:**
  - `../_base_/models/resnet18_cifar.py` (model definition)
  - `../_base_/datasets/cifar10_bs16.py` (dataset, evaluator)
  - `../_base_/schedules/cifar10_bs128.py` (training schedule)
  - `../_base_/default_runtime.py` (runtime settings)

### Data & Checkpoint
- **Data root:** `mmpretrain/data/cifar10/` (as specified in `cifar10_bs16.py`)
- **Checkpoint:** `ckpt.pth` (in working directory)

### Run Command
```bash
cd mmpretrain
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ../ckpt.pth
```

### Expected Output
The evaluation will process all 10,000 test images and print a line like:
```
accuracy/top1: 94.82
```
The reported value is the top-1 accuracy percentage on the CIFAR-10 test set.
