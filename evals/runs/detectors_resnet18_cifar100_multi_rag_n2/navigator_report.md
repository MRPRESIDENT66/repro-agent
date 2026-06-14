## Handoff: Reproducing `resnet18_cifar100` Top-1 Accuracy

### Loading Mechanism

The model `resnet18_cifar100` is registered with timm via a side-effect import. The model card's usage snippet shows:

```python
import detectors  # required side-effect: registers the custom model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```

Without `import detectors`, `timm.create_model("resnet18_cifar100", ...)` will fail because the model name is not part of the standard timm registry. The `detectors` module (present in the working directory) performs the registration when imported.

### Preprocessing

The model expects CIFAR-100-specific normalization, **not** ImageNet defaults. Read the normalization parameters from the loaded model's `pretrained_cfg`:

```python
cfg = model.pretrained_cfg
mean = cfg['mean']   # typically (0.5071, 0.4867, 0.4408)
std  = cfg['std']    # typically (0.2675, 0.2565, 0.2761)
```

Use these values for `transforms.Normalize(mean, std)`.

### Dataset

- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'` (PIL image)
- **Label field**: `'fine_label'` (100 classes; do **not** use `'coarse_label'` which has only 20 classes)
- **Loading**: offline from local cache; set `streaming=False` (default)

### Evaluation Pipeline (CPU-only)

```python
import torch
import torchvision.transforms as T
from datasets import load_dataset
import detectors  # required side-effect
import timm

# 1. Load model
model = timm.create_model("resnet18_cifar100", pretrained=True)
model.eval()

# 2. Get model-specific preprocessing
cfg = model.pretrained_cfg
transform = T.Compose([
    T.Resize(int(cfg['input_size'][1] / cfg['crop_pct'])),  # typically 36
    T.CenterCrop(cfg['input_size'][1]),                     # typically 32
    T.ToTensor(),
    T.Normalize(mean=cfg['mean'], std=cfg['std'])
])

# 3. Load test set
dataset = load_dataset("uoft-cs/cifar100", split="test")

# 4. Evaluate
correct = 0
total = 0
for example in dataset:
    img = transform(example['img']).unsqueeze(0)  # add batch dim
    label = example['fine_label']
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    correct += (pred == label)
    total += 1

accuracy = 100.0 * correct / total
print(f"Top-1 accuracy: {accuracy:.2f}%")
```

### Expected Result

The published top-1 accuracy for `resnet18_cifar100` on CIFAR-100 test set is **approximately 77.5%**. Report the computed value to two decimal places.
