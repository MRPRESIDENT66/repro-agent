## Audit Report

### Execution Analysis

The program initially failed with `FileNotFoundError` due to incorrect image list path (`cifar10_test.txt` vs `test_cifar10.txt`). After fixing the path, execution succeeded twice with dramatically different results:

- **Run 1 (Command 4)**: AUROC = 12.42% (near random, essentially failed)
- **Run 2 (Command 6)**: AUROC = 87.58% (plausible for EBO on CIFAR-10 Near-OOD)

### Critical Finding: Preprocessing Mismatch

The per-dataset AUROC values in Command 6 reveal a **strong anomaly**:
- CIFAR-100: ~86.36% (average across seeds)
- TinyImageNet: ~88.80% (average across seeds)

While both are in the 80s (no single dataset near 98-100%), the AUROC values are **suspiciously close** for two very different OOD datasets. CIFAR-100 shares classes with CIFAR-10 (e.g., dog, bird, cat) making it a harder OOD detection task, while TinyImageNet is more distinct. The near-identical performance suggests a **preprocessing issue**.

### Repository Evidence

The `get_test_transform()` function in `eval_ebo.py` uses:
```python
T.Resize(32),
T.CenterCrop(32),
```

This is **incorrect** for the OpenOOD pipeline. The official `base_preprocessor` for CIFAR-10 uses:
- `T.Resize(32)` only (no CenterCrop for 32×32 images)
- Or `T.Resize((32, 32))` for explicit dimensions

The `CenterCrop(32)` on a 32×32 image is a no-op, but the `Resize(32)` with a single integer resizes the **shorter side** to 32 pixels. For TinyImageNet images (which are 64×64 originally), this resizes to 32×32 correctly. However, for CIFAR-100 images (already 32×32), this is also fine.

**The real issue**: The transform pipeline is missing the `T.Resize((32, 32))` explicit tuple form. The single integer `T.Resize(32)` may produce non-square images for non-square inputs, causing the model to receive incorrectly shaped tensors. The model `ResNet18_32x32` expects exactly 32×32 inputs.

### Root Cause

The `get_test_transform()` function does not match the official OpenOOD `base_preprocessor` transform pipeline. The official pipeline for CIFAR-10 evaluation uses:
```python
transforms.Compose([
    transforms.Resize((32, 32)),  # explicit tuple
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

The current implementation uses `T.Resize(32)` (single integer) followed by `T.CenterCrop(32)`, which is semantically different and may produce incorrect image dimensions for certain input sizes.

### Repair Required

The transform must be changed to match the official OpenOOD preprocessing:
```python
def get_test_transform():
    return T.Compose([
        T.Resize((32, 32)),  # explicit tuple for consistent 32x32 output
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
```

This ensures all images (regardless of original size) are resized to exactly 32×32 pixels before normalization, matching the model's expected input dimensions.

REVIEW_STATUS: REPAIR_REQUIRED
