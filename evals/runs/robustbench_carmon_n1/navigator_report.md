## Handoff: Reproduce Carmon2019Unlabeled Robust Accuracy (CIFAR-10, Linf)

### 1. Load Model

```python
from robustbench.utils import load_model

model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',          # base directory
    dataset='cifar10',
    threat_model='Linf',
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'
)
model.eval()
```

**Key details:**
- `load_model` expects the checkpoint at `{model_dir}/{dataset}/{threat_model}/{model_name}.pt`
- The checkpoint is already at `robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt`
- Model architecture: `WideResNet(depth=28, widen_factor=10, sub_block1=True)`
- Model includes built-in normalization (mean/std buffers) — no external preprocessing needed

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(
    n_examples=50,
    data_dir='robustbench_data',
    transforms=None  # model handles normalization internally
)
```

**Key details:**
- `load_cifar10` returns `(torch.Tensor, torch.Tensor)` with shape `(50, 3, 32, 32)` and `(50,)`
- No external preprocessing needed — the DMWideResNet architecture applies `(x - mean) / std` internally
- Internal normalization constants: `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2471, 0.2435, 0.2616)`

### 3. AutoAttack Custom Configuration

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',
    device='cpu'
)

# Configure attacks: 1 restart each for apgd-ce and apgd-dlr
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1
adversary.apgd_targeted.n_restarts = 1  # not used but set for completeness
```

**Key details:**
- `version='custom'` enables manual attack selection
- `attacks_to_run` attribute controls which attacks execute
- `n_restarts` is set via `adversary.apgd.n_restarts` (not a constructor argument)
- CPU-only: set `device='cpu'` in constructor

### 4. Run Attack and Compute Robust Accuracy

```python
# Run attack (returns adversarial examples)
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# Compute robust accuracy
batch_size = 50
n_batches = 1  # only 50 examples
correct = 0
total = 0

with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
        x_batch = x_adv[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        outputs = model(x_batch)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

robust_acc = correct / total  # fraction (not percentage)
print(f'Robust accuracy: {robust_acc:.4f}')
```

**Key details:**
- `run_standard_evaluation` returns adversarial examples tensor
- Robust accuracy is computed as fraction (0.0–1.0), not percentage
- Model outputs logits; use `outputs.max(1)` for predictions

### 5. Complete Script

```python
import torch
from robustbench.utils import load_model
from robustbench.data import load_cifar10
from autoattack import AutoAttack

# Load model
model = load_model(
    model_name='Carmon2019Unlabeled',
    model_dir='robustbench_models',
    dataset='cifar10',
    threat_model='Linf',
    custom_checkpoint='robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt'
)
model.eval()

# Load data (first 50 examples)
x_test, y_test = load_cifar10(n_examples=50, data_dir='robustbench_data')

# Configure AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=0.031372549, version='custom', device='cpu')
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1

# Run attack
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# Compute robust accuracy
correct = (model(x_adv).argmax(1) == y_test).sum().item()
robust_acc = correct / len(x_test)
print(f'Robust accuracy: {robust_acc:.4f}')
```

### File Paths Summary
| Item | Path |
|------|------|
| Model checkpoint | `robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt` |
| CIFAR-10 data | `robustbench_data/` (standard torchvision structure) |
| RobustBench repo | `robustbench/` (installed package) |

### Expected Output
- Robust accuracy ≈ 0.60–0.65 (fraction) for first 50 examples with 1 restart each
