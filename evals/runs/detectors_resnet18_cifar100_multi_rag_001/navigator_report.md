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

The model expects CIFAR-100-specific normalization. Extract it from the loaded model's `pretrained_cfg`:

```python
cfg = model.pretrained_cfg
mean = cfg['mean']   # typically (0.5071, 0.4867, 0.4408)
std  = cfg['std']    # typically (0.2675, 0.2565, 0.2761)
```

Do **not** use ImageNet defaults. The input size is `cfg['input_size']` (expected to be `(3, 32, 32)` for CIFAR-100).

### Dataset

- **Source**: `uoft-cs/cifar100` (Hugging Face datasets)
- **Split**: `'test'` (10,000 examples)
- **Image field**: `'img'`
- **Label field**: `'fine_label'` (100 classes; do **not** use `'coarse_label'` which has only 20 classes)
- **Loading**: offline from local cache (no internet access)

### Evaluation Steps

1. **Load model** (CPU-only):
   ```python
   import detectors
   import timm
   model = timm.create_model("resnet18_cifar100", pretrained=True)
   model.eval()
   ```

2. **Get preprocessing config**:
   ```python
   cfg = model.pretrained_cfg
   data_config = timm.data.resolve_data_config(cfg)
   transform = timm.data.create_transform(**data_config)
   ```

3. **Load dataset**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("uoft-cs/cifar100", split="test")
   ```

4. **Evaluate**:
   - Apply `transform` to each `'img'`
   - Forward pass (no gradient)
   - Compare argmax logits with `'fine_label'`
   - Compute accuracy = correct / 10000 * 100

### Expected Result

The published top-1 accuracy for `resnet18_cifar100` on CIFAR-100 test set is **approximately 77.5%**. Report as a percentage with one decimal place (e.g., `77.5`).
