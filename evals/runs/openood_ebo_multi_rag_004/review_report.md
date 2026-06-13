**Audit Report**

**1. EBO Score (Faithful)**

The implementation in `openood/postprocessors/ebo_postprocessor.py` correctly computes the energy score as:
```python
conf = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
```
This matches the standard EBO formulation. The temperature parameter defaults to 1 and is configurable. The score is used as the OOD detection confidence, with higher values indicating ID samples.

**2. Preprocessing (Faithful)**

The CIFAR-10 preprocessing pipeline is consistent across both the main pipeline (`configs/preprocessors/base_preprocessor.yml` → `TestStandardPreProcessor`) and the evaluation API (`openood/evaluation_api/preprocessor.py`):
- Resize to 32×32 with bilinear interpolation
- CenterCrop to 32×32
- ToTensor()
- Normalize with CIFAR-10 mean=[0.4914, 0.4822, 0.4465] and std=[0.2470, 0.2435, 0.2616]

Both ID and OOD datasets use the same preprocessing, which is correct.

**3. Datasets (Faithful)**

The near-OOD benchmark for CIFAR-10 consists of:
- CIFAR-100 (`test_cifar100.txt`)
- TinyImageNet (`test_tin.txt`)

Far-OOD includes MNIST, SVHN, Texture, and Places365. All dataset paths and configurations match between `cifar10_ood.yml` and `openood/evaluation_api/datasets.py`.

**4. AUROC Sign Convention (Faithful)**

The sign convention is correct and consistent:
- **EBO postprocessor**: Higher energy scores indicate ID samples
- **Metrics computation** (`openood/evaluators/metrics.py`): The code explicitly negates confidence scores (`-conf`) when computing ROC curves because ID samples have higher conf values. The OOD indicator is set to 1 for OOD samples (label == -1). This correctly treats OOD as the positive class in the ROC curve, which is the standard convention.

**5. Checkpoint Set (Faithful)**

The checkpoint path is:
```
results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt
```
This matches the ResNet-18 architecture for 32×32 inputs with 10 output classes, as specified in the network config.

**6. Aggregation (Faithful)**

Metrics are reported separately for each OOD dataset (CIFAR-100, TinyImageNet, MNIST, SVHN, Texture, Places365). No averaging across datasets is performed, which is the correct behavior.

**Remaining Issues: None identified**

All components are faithful to the source code and standard conventions. The implementation correctly follows the OpenOOD framework's design for EBO-based OOD detection on CIFAR-10.
