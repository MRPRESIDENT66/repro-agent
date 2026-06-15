```python
"""
Reproduce robust accuracy of Carmon2019Unlabeled on CIFAR-10 under Linf threat model.
Uses AutoAttack custom version with ['apgd-ce', 'apgd-dlr'], 1 restart each.
Evaluates on first 50 test examples with epsilon=0.031372549.
CPU-only, offline.
"""

import torch
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from autoattack import AutoAttack

# Configuration
MODEL_NAME = "Carmon2019Unlabeled"
DATASET = BenchmarkDataset.cifar_10
THREAT_MODEL = ThreatModel.Linf
EPS = 0.031372549
N_EXAMPLES = 50
MODEL_DIR = "robustbench_models"
DATA_DIR = "robustbench_data"
DEVICE = torch.device("cpu")

# 1. Load model from pre-downloaded checkpoint
model = load_model(
    model_name=MODEL_NAME,
    model_dir=MODEL_DIR,
    dataset=DATASET,
    threat_model=THREAT_MODEL,
    device=DEVICE
)
model.eval()

# 2. Load CIFAR-10 test data (first 50 examples) with default preprocessing
#    Default preprocessing for CIFAR-10 is ToTensor() only (no normalization)
x_test, y_test = load_clean_dataset(
    dataset=DATASET,
    n_examples=N_EXAMPLES,
    data_dir=DATA_DIR,
    prepr=None  # Uses default CIFAR-10 preprocessing (ToTensor)
)

# 3. Configure AutoAttack custom version
adversary = AutoAttack(
    model,
    norm='Linf',
    eps=EPS,
    version='custom',
    device=DEVICE,
    log_path=None
)

# Set attacks and number of restarts
adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
adversary.apgd.n_restarts = 1  # APGD-CE restarts
adversary.apgd_targeted.n_restarts = 1  # APGD-DLR restarts (uses same attribute)

# 4. Generate adversarial examples
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# 5. Compute robust accuracy
with torch.no_grad():
    outputs = model(x_adv)
    _, predicted = outputs.max(1)
    correct = (predicted == y_test).sum().item()
    robust_accuracy = correct / N_EXAMPLES  # Fraction (0.0 to 1.0)

print(f"Robust accuracy: {robust_accuracy:.4f} ({robust_accuracy*100:.2f}%)")
```
