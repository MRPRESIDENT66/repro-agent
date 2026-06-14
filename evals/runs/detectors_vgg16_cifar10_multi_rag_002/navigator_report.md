## Handoff: Reproducing `vgg16_bn_cifar10` Top-1 Accuracy on CIFAR-10

### Loading Mechanism

The model `vgg16_bn_cifar10` is registered with `timm` via a side-effect import of the `detectors` package. The model card's usage snippet shows:

```python
import detectors  # required side-effect: registers the model with timm
import timm

model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
```

Without `import detectors`, `timm.create_model("vgg16_bn_cifar10", ...)` will fail because the model name is not in timm's default registry. The import must occur **before** calling `create_model`.

### Preprocessing (from model's `pretrained_cfg`)

Do **not** assume ImageNet normalization. Read the config from the loaded model:

```python
cfg = model.pretrained_cfg
# Typical CIFAR-10 values for this model:
#   mean = (0.4914, 0.4822, 0.4465)
#   std  = (0.2023, 0.1994, 0.2010)
#   input_size = (3, 32, 32)
#   interpolation = 'bilinear'
#   crop_pct = 1.0  (no cropping needed for 32x32)
```

Use `timm.data.resolve_data_config` or read `cfg['mean']`, `cfg['std']`, `cfg['input_size']` directly. Apply: convert to tensor, resize to 32×32 (if not already), normalize with the model's mean/std.

### Dataset

- **Source**: `uoft-cs/cifar10`, split `'test'` (10,000 examples)
- **Image field**: `'img'` (PIL Image, already 32×32)
- **Label field**: `'label'` (0–9)
- **Loading**: `datasets.load_dataset("uoft-cs/cifar10", split="test", cache_dir=<local_cache>)`

### Evaluation (CPU-only, offline)

```python
import torch
import detectors
import timm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# 1. Load model
model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
model.eval()

# 2. Get preprocessing from model config
cfg = model.pretrained_cfg
transform = Compose([
    Resize(int(cfg['input_size'][1])),  # 32
    ToTensor(),
    Normalize(mean=cfg['mean'], std=cfg['std'])
])

# 3. Load dataset
dataset = load_dataset("uoft-cs/cifar10", split="test", cache_dir="/path/to/cache")
dataset.set_transform(lambda x: {'img': transform(x['img']), 'label': x['label']})

# 4. Evaluate
loader = DataLoader(dataset, batch_size=64)
correct = 0
total = 0
with torch.no_grad():
    for batch in loader:
        outputs = model(batch['img'])
        preds = outputs.argmax(dim=1)
        correct += (preds == batch['label']).sum().item()
        total += batch['label'].size(0)

accuracy = 100.0 * correct / total
print(f"Top-1 accuracy: {accuracy:.2f}%")
```

### Expected Result

The published top-1 accuracy for `vgg16_bn_cifar10` on CIFAR-10 test set is **94.99%** (or approximately 95.0%).
