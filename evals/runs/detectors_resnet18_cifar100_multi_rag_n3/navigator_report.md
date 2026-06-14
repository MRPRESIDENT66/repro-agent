## Handoff: Reproduce `resnet18_cifar100` Top-1 Accuracy

### Loading Mechanism

The model `resnet18_cifar100` is registered with timm via a side-effect import. The model card shows the required loading pattern:

```python
import detectors  # side-effect: registers the custom model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```

The `import detectors` must occur **before** `timm.create_model` because the `detectors` module contains the model registration code that makes `"resnet18_cifar100"` a known model name in timm's registry. Without this import, `timm.create_model` will raise an error.

Set `model.eval()` after loading.

### Preprocessing

Do **not** assume ImageNet normalization. Read the normalization parameters from the loaded model's `pretrained_cfg`:

```python
cfg = model.pretrained_cfg
mean = cfg['mean']   # e.g., (0.4914, 0.4822, 0.4465)
std  = cfg['std']    # e.g., (0.2023, 0.1994, 0.2010)
input_size = cfg['input_size']  # e.g., (3, 32, 32)
```

The expected input size is 32×32 (CIFAR-100 native resolution). Use these values for normalization.

### Dataset

- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'` (PIL image)
- **Label field**: `'fine_label'` (0–99, 100 classes)
- **Do not use** `'coarse_label'` (20 superclasses)

Load offline from local cache:

```python
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar100", split="test", cache_dir=...)
```

### Evaluation Pipeline

1. **Transform**: Resize to `input_size` (32×32), convert to tensor, normalize with `mean`/`std` from `pretrained_cfg`.
2. **Inference**: CPU-only, no gradient, batch size manageable for CPU (e.g., 64 or 128).
3. **Metric**: Top-1 accuracy = (correct predictions / 10,000) × 100.

### Expected Output

Report a single float: the top-1 accuracy percentage (e.g., `75.23`). The model card does not state the published accuracy, so compute it from the evaluation.
