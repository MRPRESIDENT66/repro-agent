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
```

- `load_model` expects the checkpoint at `{model_dir}/{dataset}/{threat_model}/{model_name}.pt`
- The checkpoint is already at `robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt`
- Model architecture: `WideResNet(depth=28, widen_factor=10, sub_block1=True)`
- Model is returned in `eval()` mode

### 2. Load CIFAR-10 Test Data (first 50 examples)

```python
from robustbench.data import load_cifar10

x_test, y_test = load_cifar10(
    n_examples=50,
    data_dir='robustbench_data',
    transforms_test=None  # no additional transforms; model handles normalization internally
)
```

- `load_cifar10` returns `(torch.Tensor, torch.Tensor)` with shape `(50, 3, 32, 32)` and `(50,)`
- The model (DMWideResNet) applies its own normalization using `CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)` and `CIFAR10_STD = (0.2471, 0.2435, 0.2616)` via registered buffers
- No external preprocessing needed

### 3. AutoAttack Custom Version

```python
from autoattack import AutoAttack

adversary = AutoAttack(
    model,
    norm='Linf',
    eps=0.031372549,
    version='custom',
    device='cpu'
)

# Set number of restarts for each attack
adversary.apgd.n_restarts = 1
adversary.apgd_targeted = None  # not used

# Run attacks: ['apgd-ce', 'apgd-dlr']
x_adv = adversary.run_standard_evaluation(
    x_test,
    y_test,
    bs=50  # batch size; can be smaller if memory limited
)
```

- `adversary.apgd.n_restarts` controls restarts for APGD-CE and APGD-DLR
- `run_standard_evaluation` returns adversarial examples `x_adv`

### 4. Compute Robust Accuracy

```python
with torch.no_grad():
    logits = model(x_adv)
    preds = logits.argmax(dim=1)
    robust_acc = (preds == y_test).float().mean().item()

print(f"Robust accuracy: {robust_acc:.4f}")  # fraction (e.g., 0.8600)
```

- Returns fraction (0.0 to 1.0), not percentage

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

# Load data (first 50 examples)
x_test, y_test = load_cifar10(n_examples=50, data_dir='robustbench_data')

# Setup AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=0.031372549, version='custom', device='cpu')
adversary.apgd.n_restarts = 1

# Generate adversarial examples
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=50)

# Compute robust accuracy
with torch.no_grad():
    logits = model(x_adv)
    preds = logits.argmax(dim=1)
    robust_acc = (preds == y_test).float().mean().item()

print(f"Robust accuracy: {robust_acc:.4f}")
```

### Key Notes

- **CPU-only**: All tensors and models are on CPU; `device='cpu'` in AutoAttack
- **Offline**: No downloads needed; checkpoint and data are pre-downloaded
- **Expected output**: Robust accuracy ~0.86 (fraction) for Carmon2019Unlabeled on CIFAR-10 Linf with epsilon=0.031372549
