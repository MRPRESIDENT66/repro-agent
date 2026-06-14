## Handoff: Reproduce Carmon2019Unlabeled Robust Accuracy (CIFAR-10, Linf)

### 1. Load Model

```python
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',  # base directory; checkpoint at robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt
    dataset='cifar10',
    threat_model=ThreatModel.Linf,
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'  # force local path
)
model.eval()
```

**Key details:**
- `load_model` constructs path `model_dir/dataset/threat_model/model_name.pt`
- With `custom_checkpoint` set, it loads that exact file instead of downloading
- Model is a `WideResNet(depth=28, widen_factor=10, sub_block1=True)` with built-in normalization (mean/std buffers)

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(
    n_examples=50,
    data_dir='robustbench_data',
    transforms=('test',)  # standard CIFAR-10 test preprocessing (normalization to [0,1])
)
# x_test shape: (50, 3, 32, 32), values in [0, 1]
# y_test shape: (50,)
```

**Preprocessing:** Images are loaded as float32 tensors in [0, 1] range. No additional normalization is applied externally because the model handles it internally.

### 3. AutoAttack Custom Version

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,  # 8/255
    version='custom',
    device='cpu'
)

# Configure attacks: 1 restart each
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1  # sets n_restarts for APGD attacks
```

**Key attributes:**
- `adversary.attacks_to_run`: list of attack names to execute
- `adversary.apgd.n_restarts`: number of restarts for APGD (applies to both apgd-ce and apgd-dlr)

### 4. Run Attack and Compute Robust Accuracy

```python
x_adv = adversary.run_standard_evaluation(x_test, y_test)
# Returns adversarial examples tensor

# Compute robust accuracy
robust_acc = (adversary.clean_correct & adversary.correct).float().mean().item()
# Returns fraction (e.g., 0.86 for 86%)
```

**Alternative direct computation:**
```python
robust_acc = (x_adv.argmax(1) == y_test).float().mean().item()
```

### 5. Complete Script

```python
import torch
from robustbench.utils import load_model
from robustbench.data import load_cifar10
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

# Load data
x_test, y_test = load_cifar10(n_examples=50, data_dir='robustbench_data')

# Setup AutoAttack
adversary = AutoAttack(
    model, norm='Linf', eps=0.031372549,
    version='custom', device='cpu'
)
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1

# Evaluate
x_adv = adversary.run_standard_evaluation(x_test, y_test)
robust_acc = (x_adv.argmax(1) == y_test).float().mean().item()
print(f'Robust accuracy: {robust_acc:.4f}')  # fraction, e.g., 0.8600
```

**Expected output:** Fraction between 0 and 1 (e.g., ~0.86 for 86% robust accuracy).
