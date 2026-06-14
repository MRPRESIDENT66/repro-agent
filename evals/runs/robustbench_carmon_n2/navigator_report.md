## Handoff: Reproduce Carmon2019Unlabeled Robust Accuracy (CIFAR-10, Linf)

### 1. Load Model

```python
from robustbench.utils import load_model

model = load_model(
    model_name='Carmon2019Unlabeled',
    dataset='cifar10',
    threat_model='Linf',
    model_dir='robustbench_models'  # directory containing the .pt checkpoint
)
model.eval()
```

The checkpoint is expected at `robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt`.

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.data import get_preprocessing

# Get the correct preprocessing for this model
prepr = get_preprocessing(
    BenchmarkDataset.cifar_10,
    ThreatModel.Linf,
    'Carmon2019Unlabeled'
)

x_test, y_test = load_clean_dataset(
    dataset=BenchmarkDataset.cifar_10,
    n_examples=50,
    data_dir='robustbench_data',
    prepr=prepr
)
```

The preprocessing includes normalization with CIFAR-10 mean/std `(0.4914, 0.4822, 0.4465)` and `(0.2471, 0.2435, 0.2616)`.

### 3. AutoAttack Custom Version

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',
    device='cpu',
    log_path='./log.txt'
)

# Set attacks and restarts
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1  # sets n_restarts for APGD attacks

x_adv = adversary.run_standard_evaluation(x_test, y_test)
```

**Key API details:**
- `adversary.attacks_to_run` controls which attacks are executed
- `adversary.apgd.n_restarts` sets the number of restarts for APGD (both CE and DLR variants)
- `run_standard_evaluation` returns adversarial examples

### 4. Compute Robust Accuracy

```python
import torch

# Get predictions on adversarial examples
with torch.no_grad():
    outputs = model(x_adv)
    _, y_pred = outputs.max(1)

# Robust accuracy as fraction (not percentage)
robust_acc = (y_pred == y_test).float().mean().item()
print(f'Robust accuracy: {robust_acc:.4f}')  # e.g., 0.8600 for 86%
```

### Complete Script

```python
import torch
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset, get_preprocessing
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from autoattack import AutoAttack

# Load model
model = load_model('Carmon2019Unlabeled', 'cifar10', 'Linf', model_dir='robustbench_models')
model.eval()

# Load data
prepr = get_preprocessing(BenchmarkDataset.cifar_10, ThreatModel.Linf, 'Carmon2019Unlabeled')
x_test, y_test = load_clean_dataset(BenchmarkDataset.cifar_10, 50, 'robustbench_data', prepr)

# Setup AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=0.031372549, version='custom', device='cpu')
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1

# Generate adversarial examples
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# Compute robust accuracy
with torch.no_grad():
    outputs = model(x_adv)
    _, y_pred = outputs.max(1)
robust_acc = (y_pred == y_test).float().mean().item()
print(f'Robust accuracy: {robust_acc:.4f}')
```

**Expected output:** Robust accuracy as a fraction (e.g., ~0.86 for 86%).
