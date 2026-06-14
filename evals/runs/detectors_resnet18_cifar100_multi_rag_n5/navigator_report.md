## Handoff: `resnet18_cifar100` Reproduction

### Loading Mechanism
The model `resnet18_cifar100` is registered with timm via the `detectors` package. The model card's usage snippet shows:
```python
import detectors  # side-effect: registers the model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```
The `import detectors` is required **before** `timm.create_model` — it triggers the registration so that timm recognizes the model name. Without this import, `timm.create_model("resnet18_cifar100", ...)` will fail.

### Preprocessing
The model expects CIFAR-100-specific normalization. **Do not use ImageNet defaults.** Extract the normalization from the loaded model's `pretrained_cfg`:
```python
cfg = model.pretrained_cfg
mean = cfg['mean']   # typically (0.5071, 0.4867, 0.4408)
std  = cfg['std']    # typically (0.2675, 0.2565, 0.2761)
```
The input size is `cfg['input_size']` (likely `(3, 32, 32)` for CIFAR-100). Use `timm.data.create_transform` or manual `transforms.Compose` with:
- `transforms.Resize(32)` if needed (CIFAR-100 images are already 32×32)
- `transforms.ToTensor()`
- `transforms.Normalize(mean, std)`

### Dataset
- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'`
- **Label field**: `'fine_label'` (100 classes — **do not use** `'coarse_label'` which has only 20 classes)
- **Loading**: offline from local cache; set `streaming=False` (default)

### Evaluation
- **Device**: CPU only
- **Batch size**: any convenient value (e.g., 64 or 128)
- **Metric**: top-1 accuracy in percent
- **No gradient** needed: `torch.no_grad()`

### Complete Script Skeleton
```python
import detectors  # must come before timm
import timm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Load model
model = timm.create_model("resnet18_cifar100", pretrained=True)
model.eval()

# 2. Preprocessing from model's config
cfg = model.pretrained_cfg
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cfg['mean'], cfg['std']),
])

# 3. Dataset
dataset = load_dataset("uoft-cs/cifar100", split="test")
def preprocess(example):
    example['img'] = transform(example['img'])
    return example
dataset = dataset.map(preprocess, remove_columns=['coarse_label'])
dataset.set_format(type='torch', columns=['img', 'fine_label'])

# 4. DataLoader
loader = DataLoader(dataset, batch_size=128)

# 5. Evaluate
correct = 0
total = 0
with torch.no_grad():
    for batch in loader:
        images, labels = batch['img'], batch['fine_label']
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Top-1 accuracy: {accuracy:.2f}%")
```
