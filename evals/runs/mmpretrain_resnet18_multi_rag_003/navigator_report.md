## Handoff

**Entry point:** `tools/test.py`

**Config path:** `configs/resnet/resnet18_8xb16_cifar10.py`

**Inherited base configs:**
- `configs/_base_/models/resnet18_cifar.py` (model definition)
- `configs/_base_/datasets/cifar10_bs16.py` (dataset, data root `data/cifar10/`, test split, Accuracy metric with topk=(1,))
- `configs/_base_/schedules/cifar10_bs128.py` (training schedule, test_cfg)
- `configs/_base_/default_runtime.py` (default hooks, env, log level)

**Run command shape:**
```bash
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py ckpt.pth
```

**Key details:**
- `tools/test.py` accepts positional args: `config` then `checkpoint`
- The config's `test_dataloader` uses `data/cifar10/` with `split='test'` (10,000 images)
- The evaluator is `Accuracy` with `topk=(1,)` — prints `accuracy/top1:` metric
- CPU-only: no `--launcher` needed (defaults to `'none'`)

**Expected output line:** `accuracy/top1: X.XXXX` (the published top-1 accuracy on 10,000 CIFAR-10 test images)
