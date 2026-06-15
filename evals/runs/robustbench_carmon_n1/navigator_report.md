## Handoff: Reproduce Carmon2019Unlabeled Robust Accuracy (CIFAR-10, Linf)

### 1. Load Model

```python
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',          # base directory containing cifar10/Linf/
    dataset='cifar10',
    threat_model=ThreatModel.Linf,
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'  # explicit path
)
model.eval()  # already eval by default, but explicit
```

**Key details:**
- `load_model` expects `model_dir` to contain subdirectory `cifar10/Linf/` where the checkpoint `Carmon2019Unlabeled.pt` resides
- The model is a `WideResNet(depth=28, widen_factor=10, sub_block1=True)` with built-in normalization (CIFAR-10 mean/std)
- Returns model on CPU by default (`map_location=torch.device('cpu')`)

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_clean_dataset

x_test, y_test = load_clean_dataset(
    dataset='cifar10',
    n_examples=50,
    data_dir='robustbench_data'
)
```

**Preprocessing:** `transforms.ToTensor()` only (no normalization — model handles it internally via `ImageNormalizer`).

### 3. AutoAttack Custom Configuration

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',           # enables custom attack list
    attacks_to_run=['apgd-ce', 'apgd-dlr'],  # only these two attacks
    device='cpu',
    log_path='./aa_log.txt'
)

# Set number of restarts for each attack
adversary.apgd.n_restarts = 1   # APGD-CE restarts
adversary.apgd_targeted.n_restarts = 1  # APGD-DLR (uses targeted variant internally)
```

**Key attributes:**
- `adversary.apgd.n_restarts` — controls restarts for APGD-CE
- `adversary.apgd_targeted.n_restarts` — controls restarts for APGD-DLR

### 4. Run Attack and Compute Robust Accuracy

```python
# Run attack (returns adversarial examples)
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# Compute robust accuracy
from robustbench.utils import clean_accuracy

robust_acc = clean_accuracy(model, x_adv, y_test)
# Returns fraction (e.g., 0.86 for 86%)
```

### 5. Complete Script

```python
import torch
from robustbench.utils import load_model, clean_accuracy
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

# Load data (first 50 examples)
x_test, y_test = load_clean_dataset(
    dataset='cifar10',
    n_examples=50,
    data_dir='robustbench_data'
)

# Configure AutoAttack
adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',
    attacks_to_run=['apgd-ce', 'apgd-dlr'],
    device='cpu',
    log_path='./aa_log.txt'
)
adversary.apgd.n_restarts = 1
adversary.apgd_targeted.n_restarts = 1

# Evaluate
x_adv = adversary.run_standard_evaluation(x_test, y_test)
robust_acc = clean_accuracy(model, x_adv, y_test)

print(f"Robust accuracy: {robust_acc:.4f}")  # fraction, e.g., 0.8600
```

**Expected output:** Robust accuracy as a fraction (e.g., ~0.86 for 86%).
