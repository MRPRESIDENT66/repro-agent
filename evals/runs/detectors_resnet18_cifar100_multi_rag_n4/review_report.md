## Audit of `eval_detectors.py`

### Execution Summary

| Step | Command | Exit | Output |
|------|---------|------|--------|
| 1 | `py_compile` | 0 | (syntax OK) |
| 2 | Run eval | 1 | `TypeError: create_transform() got an unexpected keyword argument 'url'` |
| 3 | `py_compile` | 0 | (syntax OK after fix) |
| 4 | Run eval | 0 | `REPRO_RESULT {"metric": "top1_accuracy", "actual": 79.26, "num_examples": 10000}` |

### Error Analysis (Command 2)

The initial error was `TypeError: create_transform() got an unexpected keyword argument 'url'`. The code used `resolve_data_config(model.pretrained_cfg, model=model)` which returned a dict containing the `'url'` key (the checkpoint URL). Passing this dict directly to `create_transform(**cfg, ...)` caused the error because `create_transform` does not accept a `url` parameter.

This was fixed between Command 2 and Command 3 (likely by filtering out non-transform keys from the config dict). Command 4 succeeded.

### Correctness Verification

1. **Model registration**: ✅ `import detectors` is present before `timm.create_model("resnet18_cifar100", pretrained=True)`. Execution succeeded, confirming the model loaded with pretrained weights.

2. **Preprocessing**: ✅ The model's `pretrained_cfg` was used (not hardcoded ImageNet values). The `url` key issue was resolved.

3. **Label field**: ✅ Uses `"fine_label"` (100 classes). `"coarse_label"` is removed via `remove_columns=["coarse_label"]`.

4. **Dataset**: ✅ Uses `uoft-cs/cifar100` test split (10,000 examples). Offline mode confirmed in logs.

5. **Accuracy**: 79.26% on 10,000 examples. This is far above chance (1%) and consistent with published ResNet-18 results on CIFAR-100 (typical range ~75-79%). This confirms real model weights and correct label field.

6. **CPU-only, offline**: ✅ Environment variables set; log confirms cached dataset used.

### Conclusion

The evaluation script successfully reproduces the top-1 accuracy for `resnet18_cifar100` on CIFAR-100. The initial `url` keyword argument error was fixed, and the final run produced a valid, plausible accuracy metric.

REVIEW_STATUS: PASS
