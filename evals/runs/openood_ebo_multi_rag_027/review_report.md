REVIEW_STATUS: REPAIR_REQUIRED

**Finding: The evaluation script incorrectly constructs OOD image list paths, causing the `ImglistDataset` to receive a `num_classes` that does not match the actual label range in the OOD image list files.**

**Evidence from repository:**

1. **Public execution log (Command 4)** shows the error:
   ```
   IndexError: index 10 is out of bounds for dimension 0 with size 10
   ```
   This occurs in `openood/datasets/imglist_dataset.py` line 105 (`soft_label[sample['label']] = 1`) because the OOD image list contains labels ≥10 (e.g., CIFAR-100 labels 0-99), but the dataset was initialized with `num_classes=10`.

2. **The evaluation script** (lines 103-107) constructs the OOD image list path incorrectly:
   ```python
   ood_imglist = os.path.join(imglist_root, ood_name, f'test_{ood_name}.txt')
   # Then immediately overwrites it:
   ood_imglist = os.path.join(imglist_root, id_data_name, f'test_{ood_name}.txt')
   ```
   The first assignment is dead code; the second assignment uses `id_data_name` (cifar10) as the subdirectory, which is correct per `configs/datasets/cifar10/cifar10_ood.yml` (e.g., `./data/benchmark_imglist/cifar10/test_cifar100.txt`).

3. **However**, the script passes `ood_num_classes` based on the OOD dataset name (100 for cifar100, 200 for tin) at lines 99-103. This is **incorrect** for the OpenOOD benchmark: the OOD image list files under `cifar10/` contain labels that are **not** the original OOD dataset labels. Instead, they use label `-1` for OOD samples (as seen in `data/benchmark_imglist/osr_cifar6/test/test_cifar10_4_ood_seed1.txt`: `cifar10/test/cat/0267.png -1`).

4. **The correct approach** (used by `scripts/eval_ood.py` via `Evaluator`) is to set `num_classes=10` for OOD datasets when evaluating CIFAR-10, because the OOD image lists under `cifar10/` use label `-1` for all OOD samples, and the `ImglistDataset` creates a `soft_label` tensor of size `num_classes`. With `num_classes=10`, label `-1` causes the `IndexError` because `soft_label[-1]` is out of bounds.

5. **The public-contract audit** confirms the dataset counts are wrong: `{'cifar100': 3, 'tin': 3}` instead of `{'cifar100': 9000, 'tin': 7793}`. This is because the `IndexError` causes the dataset to skip most samples (the `BaseDataset.__getitem__` retry loop eventually exhausts valid indices), resulting in only 3 samples being successfully loaded.

**Required repair:** The OOD `ImglistDataset` must be initialized with `num_classes=10` (matching the ID dataset) and the label `-1` must be handled properly. The correct approach is to either:
- Set `num_classes=10` and handle label `-1` before the `soft_label` assignment, or
- Use the OpenOOD `Evaluator` API which handles this correctly.

The current script's `num_classes` assignment for OOD datasets (100/200) is semantically wrong for the benchmark's image list format.

REVIEW_STATUS: REPAIR_REQUIRED
