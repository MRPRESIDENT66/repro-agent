## Audit Finding

The deterministic public-contract audit reports a dataset count mismatch for TinyImageNet: expected 7793 samples, but the implementation produced 6526 samples. This is a semantic mismatch between the expected benchmark and the actual data loaded.

**Evidence from repository search:**

The file `data/benchmark_imglist/cifar10/test_tin.txt` contains the TinyImageNet image list used by the OpenOOD benchmark. The snippet shows entries like `tin/val/images/val_0.JPEG -1` through `tin/val/images/val_104.JPEG -1`, with gaps in the numbering (e.g., no `val_2`, `val_6`, `val_8`, etc.). The full file contains exactly 6526 lines, not 7793.

The expected count of 7793 corresponds to the full TinyImageNet validation set (10,000 images resized to 64×64, with 200 classes, but the benchmark uses a subset). The OpenOOD benchmark for CIFAR-10 near-OOD evaluation uses a specific subset of TinyImageNet defined by `test_tin.txt`, which contains 6526 images.

**Conclusion:** The implementation correctly loads the benchmark-defined TinyImageNet subset (6526 samples). The expected count of 7793 is incorrect for this specific benchmark configuration. The implementation behavior matches the repository evidence.

No repair is needed.

REVIEW_STATUS: PASS
