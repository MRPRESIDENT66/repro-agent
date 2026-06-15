```python
"""
Reproduce robust accuracy of Carmon2019Unlabeled on CIFAR-10 under Linf threat model
using AutoAttack custom version with ['apgd-ce', 'apgd-dlr'], 1 restart each.

Environment: CPU-only, offline. Uses pre-downloaded checkpoint and CIFAR-10 test data.
"""

import torch
import torchvision.transforms as transforms
from robustbench.utils import load_model, clean_accuracy
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from autoattack import AutoAttack

# Configuration
MODEL_NAME = "Carmon2019Unlabeled"
DATASET = BenchmarkDataset.cifar_10
THREAT_MODEL = ThreatModel.Linf
DATA_DIR = "robustbench_data"
MODEL_DIR = "robustbench_models"
N_EXAMPLES = 50
EPS = 0.031372549
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

# 2. Load CIFAR-10 test data (first 50 examples) with correct preprocessing
# Carmon2019Unlabeled uses DMWideResNet which applies its own normalization internally,
# so we only need ToTensor() for the data loader.
preprocessing = transforms.Compose([transforms.ToTensor()])
x_test, y_test = load_clean_dataset(
    dataset=DATASET,
    n_examples=N_EXAMPLES,
    data_dir=DATA_DIR,
    prepr=preprocessing
)
x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

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
adversary.apgd.n_restarts = 1  # APGD-CE uses n_restarts
adversary.apgd_targeted.n_restarts = 1  # APGD-DLR uses n_restarts (same attribute)

# 4. Run evaluation
x_adv = adversary.run_standard_evaluation(x_test, y_test)

# 5. Compute robust accuracy (fraction, not percentage)
robust_acc = clean_accuracy(model, x_adv, y_test, device=DEVICE)

print(f"Robust accuracy: {robust_acc:.4f} (fraction)")
```
