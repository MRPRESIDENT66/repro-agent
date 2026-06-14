## Handoff: Reproduce `vgg16_bn_cifar10` top-1 accuracy on CIFAR-10 test set

### Loading mechanism
The model card shows that `vgg16_bn_cifar10` is registered with timm via a side-effect import of the `detectors` module. The correct loading sequence is:

```python
import detectors   # registers the model with timm (side-effect)
import timm

model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
model.eval()
```

Without `import detectors`, `timm.create_model` will fail because the model name is not in timm's default registry.

### Preprocessing
Read the normalization from the loaded model's `pretrained_cfg` (do **not** assume ImageNet defaults). The expected transform is:

```python
from timm.data import resolve_data_config, create_transform

config = resolve_data_config(model.pretrained_cfg, model=model)
transform = create_transform(**config)
```

This will yield the correct mean, std, and resize/crop parameters that the checkpoint was trained with.

### Dataset
- **Source**: `uoft-cs/cifar10`, split `'test'` (10,000 examples)
- **Image field**: `'img'`
- **Label field**: `'label'` (10 classes)
- Load via `datasets.load_dataset("uoft-cs/cifar10", split="test")` (offline, pre-cached)

### Evaluation (CPU-only)
```python
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = load_dataset("uoft-cs/cifar10", split="test")

def preprocess(example):
    img = example['img'].convert('RGB')
    example['pixel_values'] = transform(img)
    return example

dataset = dataset.map(preprocess, remove_columns=['img'])
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

loader = DataLoader(dataset, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(loader):
        images = batch['pixel_values']
        labels = batch['label']
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Top-1 accuracy: {accuracy:.2f}%")
```

### Expected result
The published top-1 accuracy for `vgg16_bn_cifar10` on CIFAR-10 test set is **93.98%**.
