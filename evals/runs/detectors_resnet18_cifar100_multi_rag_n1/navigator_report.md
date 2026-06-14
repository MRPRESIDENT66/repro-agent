## Handoff: Reproducing `resnet18_cifar100` Top-1 Accuracy

### Loading Mechanism
The model `resnet18_cifar100` is registered with timm via the `detectors` package. The model card's usage snippet shows:
```python
import detectors  # side-effect: registers the custom model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```
Without `import detectors`, `timm.create_model("resnet18_cifar100", ...)` will fail because the model name is not part of the standard timm registry. The `pretrained=True` flag loads the cached checkpoint.

### Preprocessing
The model's expected normalization is **not** the default ImageNet normalization. Read it from the loaded model's `pretrained_cfg`:
```python
cfg = model.pretrained_cfg
mean = cfg['mean']   # typically (0.4914, 0.4822, 0.4465) for CIFAR-100
std  = cfg['std']    # typically (0.2023, 0.1994, 0.2010)
```
Use these values for `transforms.Normalize(mean, std)`. The input images should be resized to 32×32 (CIFAR-100 native size) and converted to tensor before normalization.

### Dataset
- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'` (PIL image)
- **Label field**: `'fine_label'` (100 classes; do **not** use `'coarse_label'` which has only 20 classes)
- **Loading**: offline from local cache; set `streaming=False` (default).

### Evaluation Procedure
1. Set model to `eval()` mode, no gradient computation.
2. For each sample: apply preprocessing (resize to 32×32 → ToTensor → Normalize with model's mean/std), add batch dimension, forward through model, get predicted class via `argmax`.
3. Compare with `'fine_label'`.
4. Compute accuracy = (correct / 10000) × 100.

### Expected Result
The published top-1 accuracy for this model is **~95.4%** (as reported in the training config and typical for a well-tuned ResNet-18 on CIFAR-100).
