REVIEW_STATUS: REPAIR_REQUIRED

**Finding: Dataset count mismatch for TinyImageNet**

The deterministic public-contract audit reports that the expected TinyImageNet dataset count is 7793, but the implementation produced 6526. This is a semantic mismatch between the implementation and the authoritative reference.

**Repository evidence:**

1. The reference script `scripts/eval_ood.py` uses the `Evaluator` class from `openood/evaluation_api`, which automatically handles dataset selection and preprocessing for Near-OOD evaluation. The implementation instead manually constructs `ImglistDataset` with hardcoded paths.

2. The implementation uses `data/benchmark_imglist/cifar100/test_tin.txt` as the TinyImageNet image list. The retrieved snippet shows this file contains entries like `tin/val/images/val_994.JPEG -1` with paths under `tin/val/images/`. The implementation sets `OOD_TIN_DATA_DIR = './data/images_classic'`, which would resolve to `./data/images_classic/tin/val/images/val_994.JPEG`.

3. The expected count of 7793 corresponds to the full TinyImageNet validation set (10,000 images minus 2,207 that are part of the training set). The implementation's count of 6526 indicates it is using a different or incomplete image list.

4. The `Evaluator` class in `scripts/eval_ood.py` uses `data_root=os.path.join(ROOT_DIR, 'data')` and `config_root=os.path.join(ROOT_DIR, 'configs')`, which allows it to resolve the correct dataset configuration and image lists from the OpenOOD configs. The implementation bypasses this mechanism.

**Impact:** The AUROC computation uses a different (smaller) OOD dataset than the reference, producing an incorrect metric value. The reported `actual: 92.41` cannot be verified against the reference because the dataset composition is wrong.

**Required repair:** The implementation must use the same TinyImageNet dataset as the reference `scripts/eval_ood.py`. This means either:
- Using the `Evaluator` class directly (as the reference does), or
- Identifying and using the correct image list that produces 7793 TinyImageNet samples, matching the reference behavior.

REVIEW_STATUS: REPAIR_REQUIRED
