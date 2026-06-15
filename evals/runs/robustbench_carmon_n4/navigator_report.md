## Handoff: Reproduce Carmon2019Unlabeled Robust Accuracy (CIFAR-10, Linf)

### 1. Load Model

```python
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',          # base directory
    dataset='cifar10',
    threat_model=ThreatModel.Linf,
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'  # explicit path
)
model.eval()
```

**Key details:**
- `load_model` expects `model_dir` as base; it constructs path as `{model_dir}/{dataset}/{threat_model.value}/{model_name}.pt`
- The checkpoint at `robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt` will be loaded directly (no download)
- Model is a `WideResNet(depth=28, widen_factor=10, sub_block1=True)` with built-in normalization (mean/std buffers)

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_clean_dataset

x_test, y_test = load_clean_dataset(
    n_examples=50,
    dataset='cifar10',
    data_dir='robustbench_data',
    threat_model=ThreatModel.Linf
)
```

**Preprocessing:** No additional transforms applied; the model's internal normalization handles `CIFAR10_MEAN=(0.4914, 0.4822, 0.4465)` and `CIFAR10_STD=(0.2471, 0.2435, 0.2616)`.

### 3. AutoAttack Custom Version

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',           # enables custom attack list
    attacks_to_run=['apgd-ce', 'apgd-dlr'],
    device='cpu',
    log_path=None
)
adversary.apgd.n_restarts = 1  # set n_restarts for APGD
```

**API details:**
- `version='custom'` allows specifying `attacks_to_run`
- `adversary.apgd.n_restarts = 1` sets restarts for both APGD variants (they share the same APGDAttack instance)
- Attacks run sequentially: `['apgd-ce', 'apgd-dlr']`

### 4. Run Attack and Compute Robust Accuracy

```python
x_adv = adversary.run_standard_evaluation(x_test, y_test)
robust_acc = (x_adv.max(1)[1] == y_test).float().mean().item()
```

**Return format:** Fraction (0.0 to 1.0), not percentage.

### 5. Complete Script

```python
import torch
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import ThreatModel
from autoattack import AutoAttack

# Load model
model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',
    dataset='cifar10',
    threat_model=ThreatModel.Linf,
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'
)
model.eval()

# Load data (first 50 examples)
x_test, y_test = load_clean_dataset(
    n_examples=50,
    dataset='cifar10',
    data_dir='robustbench_data',
    threat_model=ThreatModel.Linf
)

# Setup AutoAttack
adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',
    attacks_to_run=['apgd-ce', 'apgd-dlr'],
    device='cpu',
    log_path=None
)
adversary.apgd.n_restarts = 1

# Evaluate
x_adv = adversary.run_standard_evaluation(x_test, y_test)
robust_acc = (x_adv.max(1)[1] == y_test).float().mean().item()
print(f"Robust accuracy: {robust_acc:.4f}")
```

**Expected output:** Fraction between 0 and 1 (e.g., 0.8600 for 86% robust accuracy).
