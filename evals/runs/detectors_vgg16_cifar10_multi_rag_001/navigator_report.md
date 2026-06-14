## Handoff

**Model loading**  
The model card shows that `vgg16_bn_cifar10` is registered with timm via a side-effect import of the `detectors` package. The correct loading sequence is:

```python
import detectors   # registers the custom model into timm's registry
import timm

model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
model.eval()
```

Without `import detectors`, `timm.create_model` will fail because the model name is not part of the standard timm distribution.

**Preprocessing**  
Do **not** assume ImageNet normalization. Read the model's own `pretrained_cfg` after loading:

```python
cfg = model.pretrained_cfg
mean = cfg['mean']          # e.g. (0.4914, 0.4822, 0.4465)
std  = cfg['std']           # e.g. (0.2023, 0.1994, 0.2010)
input_size = cfg['input_size']  # e.g. (3, 32, 32)
```

Use these values for `transforms.Normalize(mean, std)` and resize/crop to `input_size[1:]` (32×32 for CIFAR-10). The model expects raw uint8 images (0–255) converted to float and then normalized.

**Dataset**  
- Source: `uoft-cs/cifar10`, split `'test'` (10 000 examples).  
- Image field: `'img'` (PIL image).  
- Label field: `'label'` (0–9).  
- Load with `datasets.load_dataset("uoft-cs/cifar10", split="test")`.

**Environment**  
- CPU-only, offline. Use `model.to("cpu")` (already default).  
- No GPU, no internet. All weights and dataset are pre-cached.

**Evaluation**  
- Batch inference (e.g., batch size 64 or 128) with `torch.no_grad()`.  
- Compute top-1 accuracy: `correct / total * 100`.  
- Report as a single float percentage (e.g., `93.45`).
