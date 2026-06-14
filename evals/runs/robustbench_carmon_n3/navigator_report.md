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
# load_model() automatically looks for the checkpoint in MODEL_DIR/<dataset>/<threat_model>/<model_name>.pt
model = load_model(
    model_name=MODEL_NAME,
    dataset=DATASET,
    threat_model=THREAT_MODEL,
    model_dir=MODEL_DIR,
    device=DEVICE,
)
model.eval()

# ── 2. Load CIFAR-10 test data (first 50 examples) ────────────────────────
# The preprocessing for Carmon2019Unlabeled under Linf is identity (no normalization)
# because the model architecture (DMWideResNet) applies its own normalization internally.
# We pass transforms.ToTensor() as the preprocessing.
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
    log_path=None,              # no logging to disk
)

# Set the attacks and number of restarts via the `attacks_to_run` attribute
adversary.attacks_to_run = ["apgd-ce", "apgd-dlr"]

# Set number of restarts for each attack via the `apgd` attribute
adversary.apgd.n_restarts = 1   # applies to both apgd-ce and apgd-dlr

# ── 4. Run evaluation ──────────────────────────────────────────────────────
# run_standard_evaluation returns adversarial examples and also prints robust accuracy
x_adv = adversary.run_standard_evaluation(
    x_test,
    y_test,
    bs=50,                      # batch size = number of examples
)

# ── 5. Compute and print robust accuracy ───────────────────────────────────
# robust accuracy is computed as fraction (not percentage) by AutoAttack
# We can also compute it manually:
with torch.no_grad():
    logits = model(x_adv)
    preds = logits.argmax(dim=1)
    robust_acc = (preds == y_test).float().mean().item()

print(f"Robust accuracy (fraction): {robust_acc:.4f}")
print(f"Robust accuracy (percentage): {robust_acc * 100:.2f}%")
```
