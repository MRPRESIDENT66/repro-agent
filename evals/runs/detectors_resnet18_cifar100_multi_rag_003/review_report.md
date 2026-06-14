## Audit of `eval_detectors.py`

### Execution Result
- **Status**: Success (exit=0)
- **Accuracy**: 79.26% on 10,000 CIFAR-100 test examples
- **Expected**: ~77.6% (per model card)

### Code Correctness Verification

| Requirement | Implementation | Status |
|---|---|---|
| Register model via `import detectors` | `import detectors  # noqa: F401` | ✅ |
| Load model with `timm.create_model("resnet18_cifar100", pretrained=True)` | Yes | ✅ |
| Use `model.pretrained_cfg` for mean/std/input_size/interpolation | Yes | ✅ |
| Dataset: `uoft-cs/cifar100`, split=`test` | Yes | ✅ |
| Image field: `'img'` | Yes | ✅ |
| Label field: `'fine_label'` (not `'coarse_label'`) | Yes | ✅ |
| CPU eval mode, no grad | Yes | ✅ |
| Top-1 accuracy computation | Yes | ✅ |

### Analysis
- The accuracy 79.26% is well above chance (1%) and consistent with a real ResNet-18 on CIFAR-100.
- The slight deviation from the stated ~77.6% is within normal variation for different training runs/hyperparameters.
- The code correctly uses the model's own normalization parameters (not ImageNet defaults).
- All 10,000 test examples were evaluated.
- No errors, warnings (aside from unrelated deprecation warning), or suspicious behavior.

### Conclusion
The implementation faithfully reproduces the model card's usage instructions and evaluation protocol. No repair is needed.

REVIEW_STATUS: PASS
