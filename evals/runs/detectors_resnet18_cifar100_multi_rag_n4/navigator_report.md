## Handoff: Reproducing `resnet18_cifar100` Top-1 Accuracy

### Loading Mechanism

The model `resnet18_cifar100` is registered with timm via a side-effect import of the `detectors` module. The model card's usage snippet shows:

```python
import detectors  # required side-effect: registers the custom model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```

Without `import detectors`, `timm.create_model("resnet18_cifar100", ...)` will fail because the model name is not in timm's default registry. The `pretrained=True` flag loads the cached checkpoint weights.

### Preprocessing

Do **not** assume ImageNet normalization. Read the preprocessing configuration from the loaded model's `pretrained_cfg`:

```python
cfg = model.pretrained_cfg
# cfg contains: mean, std, input_size, interpolation, crop_pct, etc.
```

The expected normalization (mean, std) and input size are model-specific and stored in this config. Use `timm.data.create_transform` or manually apply:

```python
from timm.data import create_transform
transform = create_transform(**cfg, is_training=False)
```

This will apply the correct resize, center crop, ToTensor, and normalization.

### Dataset

- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'`
- **Label field**: `'fine_label'` (100 classes — do **not** use `'coarse_label'` which has only 20 classes)
- **Loading**: offline from local cache (no internet connection)

```python
from datasets import load_dataset
dataset = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)
```

### Evaluation

- **Device**: CPU only
- **Batch size**: any reasonable value (e.g., 64 or 128)
- **Metric**: top-1 accuracy in percent

### Complete Script Outline

```python
import torch
import detectors  # required for model registration
import timm
from timm.data import create_transform
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load model
model = timm.create_model("resnet18_cifar100", pretrained=True)
model.eval()

# Get preprocessing from model config
cfg = model.pretrained_cfg
transform = create_transform(**cfg, is_training=False)

# Load dataset
dataset = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)

# Apply transform
def preprocess(example):
    example["img"] = transform(example["img"])
    return example

dataset = dataset.map(preprocess, remove_columns=["coarse_label"])
dataset.set_format(type="torch", columns=["img", "fine_label"])

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        images, labels = batch["img"], batch["fine_label"]
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Top-1 accuracy: {accuracy:.2f}%")
```

### Key Points

1. **Must import `detectors`** before `timm.create_model` — this is the registration side-effect.
2. **Preprocessing comes from `model.pretrained_cfg`**, not hardcoded ImageNet values.
3. **Use `fine_label`** (100 classes), not `coarse_label`.
4. **CPU-only, offline** — ensure all data and weights are cached locally.
