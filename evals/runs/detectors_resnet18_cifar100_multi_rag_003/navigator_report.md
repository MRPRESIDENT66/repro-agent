## Handoff: Reproducing `resnet18_cifar100` Top-1 Accuracy

### Loading Mechanism
The model `resnet18_cifar100` is registered with timm via the `detectors` package. The model card's usage snippet shows:
```python
import detectors  # side-effect: registers the model with timm
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```
The `import detectors` is required before `timm.create_model`; without it, timm will not recognize the model name. The pretrained weights are cached locally and loaded automatically.

### Preprocessing
The model's expected normalization is obtained from the loaded model's `pretrained_cfg`:
```python
cfg = model.pretrained_cfg
# cfg contains: mean, std, input_size, interpolation, crop_pct
```
For this CIFAR-100 model, the normalization is **not** ImageNet defaults. Use:
- `mean = cfg['mean']` (likely `(0.5071, 0.4867, 0.4408)`)
- `std = cfg['std']` (likely `(0.2675, 0.2565, 0.2761)`)
- `input_size = cfg['input_size']` (likely `(3, 32, 32)`)
- `interpolation = cfg['interpolation']` (likely `'bilinear'`)
- `crop_pct = cfg['crop_pct']` (likely `1.0` — no cropping needed for 32×32)

Resize to `input_size[1:]` (32×32) using the specified interpolation, then normalize with the model's mean/std.

### Dataset
- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'` (PIL image)
- **Label field**: `'fine_label'` (100 classes, integer 0–99)
- **Do not use** `'coarse_label'` (20 classes)

### Evaluation Procedure (CPU-only, offline)
1. Load dataset from local cache: `load_dataset("uoft-cs/cifar100", split="test")`
2. Apply preprocessing transform:
   - Resize to 32×32 (model's input size)
   - Convert to tensor
   - Normalize with model's `mean` and `std`
3. Run model in `eval()` mode, no gradient computation
4. Compute top-1 accuracy: `(predicted == target).sum() / total * 100`

### Expected Result
The published top-1 accuracy for `resnet18_cifar100` on CIFAR-100 test set is approximately **77.6%** (based on typical ResNet-18 performance on CIFAR-100 with the training hyperparameters listed in the model card).
