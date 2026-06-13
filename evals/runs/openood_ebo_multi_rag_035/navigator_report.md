To reproduce the official EBO Near-OOD AUROC for CIFAR-10 using the s0, s1, and s2 CrossEntropy ResNet-18 checkpoints with CIFAR-100 and TinyImageNet as Near-OOD datasets, follow these grounded steps based on the OpenOOD repository structure and evaluation semantics:

---

### **1. Checkpoint Paths**
The official CIFAR-10 ResNet-18 CrossEntropy checkpoints are located at:
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt`
- `results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt`

These correspond to seeds 0, 1, and 2 (s0, s1, s2).

---

### **2. Evaluation Command**
Use the unified evaluator `scripts/eval_ood.py`, which aggregates results across all OOD datasets defined in `configs/datasets/cifar10/cifar10_ood.yml`—including **CIFAR-100** and **TinyImageNet** as Near-OOD.

For each seed, run:
```bash
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor ebo \
    --save-score --save-csv
```

This script:
- Automatically detects subfolders `s0`, `s1`, `s2`.
- Loads the `best.ckpt` from each.
- Evaluates using the **EBO postprocessor** (Energy-Based Outlier detection).
- Computes AUROC against all OOD datasets listed in `cifar10_ood.yml`, which includes:
  - `cifar100` (Near-OOD)
  - `tin` (TinyImageNet, resized to 32×32, Near-OOD)

> ✅ **Preserves repository semantics**: Uses the same data loaders, preprocessing (`base_preprocessor.yml`), and OOD protocol as the official scripts.

---

### **3. EBO Semantics**
- **EBO** uses the energy score:  
  \( E(\mathbf{x}) = -T \cdot \log \sum_{i=1}^{K} e^{f_i(\mathbf{x}) / T} \)  
  where \( T = 1 \) by default (as seen in `cifar10_test_ood_ebo.sh`).
- Higher energy → more likely OOD.
- AUROC is computed by comparing ID (CIFAR-10 test set) vs. each OOD dataset separately.

---

### **4. Data & Preprocessing**
- **ID Data**: CIFAR-10 test set (`./data/benchmark_imglist/cifar10/test_cifar10.txt`)
- **Near-OOD**:
  - **CIFAR-100**: `./data/benchmark_imglist/cifar10/cifar100_test.txt`
  - **TinyImageNet (tin)**: `./data/benchmark_imglist/cifar10/tin_test.txt` (32×32 version)
- **Preprocessing**: Standardization with CIFAR-10 mean/std, no augmentation at test time (`base_preprocessor.yml`).

---

### **5. Aggregation**
- The `eval_ood.py` script outputs per-seed and **mean ± std** AUROC across s0/s1/s2.
- Report the **mean Near-OOD AUROC** averaged over:
  - CIFAR-100 AUROC
  - TinyImageNet AUROC  
  (as done in OpenOOD benchmark tables)

> 📌 Example output format in CSV:
> ```
> postprocessor, id_dataset, ood_dataset, auroc
> ebo, cifar10, cifar100, 89.2
> ebo, cifar10, tin, 87.5
> ```

Final reported metric: **mean of these two AUROCs**, averaged over 3 seeds, in **percentage** (e.g., `88.35`).

---

### **6. CPU & Offline Execution**
- Set `--num_workers 0` if DataLoader hangs on CPU.
- Ensure `torch.set_num_threads(1)` or limit via environment (`OMP_NUM_THREADS=1`) to avoid CPU overload.
- No internet access needed—all data and configs are local.

---

### **Final Handoff**
Run the above `eval_ood.py` command. Extract the **mean AUROC (%)** for **CIFAR-100** and **TinyImageNet** across seeds s0–s2. Average these two values to obtain the official **EBO Near-OOD AUROC** for CIFAR-10.
