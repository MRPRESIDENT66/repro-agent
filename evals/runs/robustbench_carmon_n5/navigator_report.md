```python
"""
Reproduce robust accuracy of Carmon2019Unlabeled on CIFAR-10 under Linf threat model
using AutoAttack custom version with ['apgd-ce', 'apgd-dlr'], 1 restart each.

Environment: CPU-only, offline.
Checkpoint: robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt
Data: robustbench_data (first 50 CIFAR-10 test examples)
Epsilon: 0.031372549
"""

import torch
import torchvision.transforms as transforms
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from autoattack import AutoAttack

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "Carmon2019Unlabeled"
DATASET = BenchmarkDataset.cifar_10
THREAT_MODEL = ThreatModel.Linf
EPS = 0.031372549
N_EXAMPLES = 50
MODEL_DIR = "robustbench_models"
DATA_DIR = "robustbench_data"
DEVICE = torch.device("cpu")

# ── 1. Load model from local checkpoint ────────────────────────────────────
# load_model() automatically looks for the checkpoint at:
#   robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt
model = load_model(
    model_name=MODEL_NAME,
    model_dir=MODEL_DIR,
    dataset=DATASET,
    threat_model=THREAT_MODEL,
    device=DEVICE,
)
model.eval()

# ── 2. Load CIFAR-10 test data (first 50 examples) ────────────────────────
# The preprocessing used by Carmon2019Unlabeled is the default identity
# (no extra transforms) because the model's DMWideResNet normalizes internally
# using its own mean/std buffers.
preprocessing = transforms.Compose([transforms.ToTensor()])

x_test, y_test = load_clean_dataset(
    dataset=DATASET,
    n_examples=N_EXAMPLES,
    data_dir=DATA_DIR,
    prepr=preprocessing,
)

# ── 3. Set up AutoAttack custom version ────────────────────────────────────
adversary = AutoAttack(
    model,
    norm="Linf",
    eps=EPS,
    version="custom",           # enables custom attack list
    device=DEVICE,
    log_path=None,              # suppress logging
)

# Set the attacks and number of restarts via the `attacks_to_run` attribute
adversary.attacks_to_run = ["apgd-ce", "apgd-dlr"]

# Set n_restarts for each APGD attack via the `apgd` attribute
adversary.apgd.n_restarts = 1

# ── 4. Run evaluation ──────────────────────────────────────────────────────
x_adv = adversary.run_standard_evaluation(
    x_test,
    y_test,
    bs=50,                      # batch size = number of examples
)

# ── 5. Compute robust accuracy ─────────────────────────────────────────────
with torch.no_grad():
    outputs = model(x_adv)
    _, predicted = outputs.max(1)
    correct = (predicted == y_test).sum().item()
    robust_acc = correct / N_EXAMPLES   # fraction (not percentage)

print(f"Robust accuracy: {robust_acc:.4f} ({100 * robust_acc:.2f}%)")
```
