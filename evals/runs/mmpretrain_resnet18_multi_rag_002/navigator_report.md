## Handoff

**Evaluation entry point:**  
`mmpretrain/tools/test.py`

**Config path:**  
`mmpretrain/configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model definition)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset settings, batch size 16, test split with `Accuracy` evaluator)
- `configs/_base_/schedules/cifar10_bs128.py` (training schedule, not used for evaluation)
- `configs/_base_/default_runtime.py` (runtime defaults)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**Key details:**
- The script accepts `config` and `checkpoint` as positional arguments (lines 17-18 of `tools/test.py`)
- The config sets `test_dataloader` to use `data/cifar10/` with `split='test'` and batch size 16
- The evaluator is `Accuracy` with `topk=(1,)`, which prints `accuracy/top1:` in the output
- The CIFAR-10 test set contains 10,000 images
- The environment is CPU-only, so `--launcher none` (the default) is appropriate
