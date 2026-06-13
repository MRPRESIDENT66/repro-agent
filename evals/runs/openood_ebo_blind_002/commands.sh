ls -la
ls -la results/
ls -la data/
cat openood/evaluation_api/evaluator.py
ls -la results/cifar10_resnet18_32x32_base_e100_lr0.1_default/
ls -la data/benchmark_imglist/
ls -la data/images_classic/
ls -la results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/
ls -la data/benchmark_imglist/cifar10/
ls -la data/images_classic/cifar10/
cat data/benchmark_imglist/cifar10/test_cifar100.txt | head -5
cat data/benchmark_imglist/cifar10/test_tin.txt | head -5
cat data/benchmark_imglist/cifar10/test_cifar10.txt | head -5
cat openood/evaluation_api/evaluator.py
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluation_api/datasets.py
cat openood/evaluation_api/postprocessor.py
cat openood/postprocessors/__init__.py
cat openood/postprocessors/ebo.py
cat openood/postprocessors/ebo_postprocessor.py
cat openood/pipelines/test_ood_pipeline.py
cat openood/evaluators/__init__.py
cat openood/evaluators/ood_evaluator.py
cat openood/evaluators/metrics.py
cat scripts/ood/residual/cifar10_test_ood_residual.sh
ls openood/configs/
ls configs/
cat configs/datasets/cifar10/cifar10.yml
cat configs/datasets/cifar10/cifar10_ood.yml
cat configs/networks/resnet18_32x32.yml
cat configs/pipelines/test/test_ood.yml
cat configs/preprocessors/base_preprocessor.yml
cat configs/postprocessors/ebo.yml
cat openood/datasets/__init__.py
cat openood/datasets/imglist_dataset.py
cat openood/datasets/utils.py
cat openood/preprocessors/test_preprocessor.py
cat openood/preprocessors/utils.py
cat openood/preprocessors/base_preprocessor.py
cat openood/preprocessors/transform.py
cat openood/preprocessors/test_preprocessor.py
cat openood/networks/__init__.py
cat openood/networks/resnet18_32x32.py
cat openood/networks/utils.py
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/datasets.py
python -c "
import torch
ckpt = torch.load('results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt', map_location='cpu')
print(type(ckpt))
if isinstance(ckpt, dict):
    print(list(ckpt.keys()))
"
cat > eval_ebo.py <<'PYEOF'
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Add repo root to path
sys.path.insert(0, '.')

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.transform import normalization_dict
from openood.evaluation_api.evaluator import DATA_INFO

import torchvision.transforms as tvs_trans

# Configuration
data_root = './data'
num_classes = 10

# Normalization for CIFAR-10
norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2470, 0.2435, 0.2616]

# Preprocessing: same as TestStandardPreProcessor
transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=norm_mean, std=norm_std),
])

# ID test dataset
id_dataset = ImglistDataset(
    name='cifar10_test',
    imglist_pth=os.path.join(data_root, 'benchmark_imglist/cifar10/test_cifar10.txt'),
    data_dir=os.path.join(data_root, 'images_classic'),
    num_classes=num_classes,
    preprocessor=transform,
)

id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0)

# OOD datasets
ood_datasets = {
    'cifar100': {
        'imglist': 'benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': 'images_classic',
    },
    'tin': {
        'imglist': 'benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': 'images_classic',
    },
}

ood_loaders = {}
for name, cfg in ood_datasets.items():
    dataset = ImglistDataset(
        name=f'cifar10_ood_{name}',
        imglist_pth=os.path.join(data_root, cfg['imglist']),
        data_dir=os.path.join(data_root, cfg['data_dir']),
        num_classes=num_classes,
        preprocessor=transform,
    )
    ood_loaders[name] = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# Model setup
def load_model(ckpt_path):
    model = ResNet18_32x32(num_classes=num_classes)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_energy(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC using sklearn"""
    from sklearn import metrics
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Higher energy = more ID-like (in-distribution has higher energy typically)
    # For EBO: ID samples have higher energy, OOD have lower energy
    # So we want AUROC for "ID vs OOD" - higher score = ID
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc * 100  # percentage

def evaluate_run(ckpt_path, run_name):
    print(f"\n=== Evaluating {run_name} ===")
    model = load_model(ckpt_path)
    
    # Get ID scores
    id_energy = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data']
            logits = model(data)
            energy = compute_energy(logits)
            id_energy.extend(energy.cpu().numpy().tolist())
    id_energy = np.array(id_energy)
    
    results = {}
    for ood_name, loader in ood_loaders.items():
        ood_energy = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data']
                logits = model(data)
                energy = compute_energy(logits)
                ood_energy.extend(energy.cpu().numpy().tolist())
        ood_energy = np.array(ood_energy)
        
        auroc = compute_auroc(id_energy, ood_energy)
        results[ood_name] = auroc
        print(f"  {ood_name}: AUROC = {auroc:.2f}%")
    
    return results

# Run all three seeds
seeds = ['s0', 's1', 's2']
base_path = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'

all_results = {}
for seed in seeds:
    ckpt_path = os.path.join(base_path, seed, 'best.ckpt')
    all_results[seed] = evaluate_run(ckpt_path, seed)

# Compute aggregate
print("\n\n=== AGGREGATE RESULTS ===")
dataset_names = ['cifar100', 'tin']
run_metrics = {}
for seed in seeds:
    run_metrics[seed] = {d: all_results[seed][d] for d in dataset_names}

# Dataset mean per run, then mean across runs
run_means = []
for seed in seeds:
    run_mean = np.mean([all_results[seed][d] for d in dataset_names])
    run_means.append(run_mean)
    print(f"{seed}: dataset_mean = {run_mean:.4f}")

actual = np.mean(run_means)
print(f"\nFinal AUROC (dataset_mean_then_run_mean): {actual:.4f}%")

# Print the evidence line
import json
evidence = {
    "metric": "near_ood_auroc",
    "actual": float(actual),
    "datasets": {
        "cifar100": len(ood_loaders['cifar100'].dataset),
        "tin": len(ood_loaders['tin'].dataset),
    },
    "run_metrics": run_metrics,
    "aggregation": "dataset_mean_then_run_mean"
}
print(f"\nREPRO_RESULT {json.dumps(evidence)}")
PYEOF
python eval_ebo.py
cat > eval_ebo.py <<'PYEOF'
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from sklearn import metrics

# Add repo root to path
sys.path.insert(0, '.')

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

import torchvision.transforms as tvs_trans

# Configuration
data_root = './data'
num_classes = 10

# Normalization for CIFAR-10
norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2470, 0.2435, 0.2616]

# Preprocessing: same as TestStandardPreProcessor
transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=norm_mean, std=norm_std),
])

# ID test dataset
id_dataset = ImglistDataset(
    name='cifar10_test',
    imglist_pth=os.path.join(data_root, 'benchmark_imglist/cifar10/test_cifar10.txt'),
    data_dir=os.path.join(data_root, 'images_classic'),
    num_classes=num_classes,
    preprocessor=transform,
)

id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0)

# OOD datasets
ood_datasets = {
    'cifar100': {
        'imglist': 'benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': 'images_classic',
    },
    'tin': {
        'imglist': 'benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': 'images_classic',
    },
}

ood_loaders = {}
for name, cfg in ood_datasets.items():
    dataset = ImglistDataset(
        name=f'cifar10_ood_{name}',
        imglist_pth=os.path.join(data_root, cfg['imglist']),
        data_dir=os.path.join(data_root, cfg['data_dir']),
        num_classes=num_classes,
        preprocessor=transform,
    )
    ood_loaders[name] = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# Model setup
def load_model(ckpt_path):
    model = ResNet18_32x32(num_classes=num_classes)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_energy(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC using sklearn"""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Higher energy = more ID-like (in-distribution has higher energy typically)
    # For EBO: ID samples have higher energy, OOD have lower energy
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc * 100  # percentage

def evaluate_run(ckpt_path, run_name):
    print(f"\n=== Evaluating {run_name} ===")
    model = load_model(ckpt_path)
    
    # Get ID scores
    id_energy = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data']
            logits = model(data)
            energy = compute_energy(logits)
            id_energy.extend(energy.cpu().numpy().tolist())
    id_energy = np.array(id_energy)
    
    results = {}
    for ood_name, loader in ood_loaders.items():
        ood_energy = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data']
                logits = model(data)
                energy = compute_energy(logits)
                ood_energy.extend(energy.cpu().numpy().tolist())
        ood_energy = np.array(ood_energy)
        
        auroc = compute_auroc(id_energy, ood_energy)
        results[ood_name] = auroc
        print(f"  {ood_name}: AUROC = {auroc:.2f}%")
    
    return results

# Run all three seeds
seeds = ['s0', 's1', 's2']
base_path = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'

all_results = {}
for seed in seeds:
    ckpt_path = os.path.join(base_path, seed, 'best.ckpt')
    all_results[seed] = evaluate_run(ckpt_path, seed)

# Compute aggregate
print("\n\n=== AGGREGATE RESULTS ===")
dataset_names = ['cifar100', 'tin']
run_metrics = {}
for seed in seeds:
    run_metrics[seed] = {d: all_results[seed][d] for d in dataset_names}

# Dataset mean per run, then mean across runs
run_means = []
for seed in seeds:
    run_mean = np.mean([all_results[seed][d] for d in dataset_names])
    run_means.append(run_mean)
    print(f"{seed}: dataset_mean = {run_mean:.4f}")

actual = np.mean(run_means)
print(f"\nFinal AUROC (dataset_mean_then_run_mean): {actual:.4f}%")

# Print the evidence line
evidence = {
    "metric": "near_ood_auroc",
    "actual": float(actual),
    "datasets": {
        "cifar100": len(ood_loaders['cifar100'].dataset),
        "tin": len(ood_loaders['tin'].dataset),
    },
    "run_metrics": run_metrics,
    "aggregation": "dataset_mean_then_run_mean"
}
print(f"\nREPRO_RESULT {json.dumps(evidence)}")
PYEOF
python eval_ebo.py
cat openood/datasets/imglist_dataset.py
cat openood/datasets/utils.py
cat openood/preprocessors/test_preprocessor.py
cat openood/preprocessors/base_preprocessor.py
cat openood/datasets/utils.py
cat openood/preprocessors/test_preprocessor.py
cat openood/preprocessors/base_preprocessor.py
cat openood/datasets/utils.py
cat openood/preprocessors/test_preprocessor.py
cat openood/preprocessors/base_preprocessor.py
cat openood/datasets/utils.py
cat openood/preprocessors/test_preprocessor.py
cat > eval_ebo.py <<'PYEOF'
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

sys.path.insert(0, '.')

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.utils.config import Config

# Load configs
dataset_config = Config.from_yaml('configs/datasets/cifar10/cifar10.yml')
ood_config = Config.from_yaml('configs/datasets/cifar10/cifar10_ood.yml')
preprocessor_config = Config.from_yaml('configs/preprocessors/base_preprocessor.yml')

# Merge configs into a single config object
class MergedConfig:
    def __init__(self):
        self.dataset = dataset_config['dataset']
        self.preprocessor = preprocessor_config['preprocessor']
        # Add normalization type info
        self.dataset['normalization_type'] = 'cifar10'
        self.dataset['pre_size'] = 32
        self.dataset['image_size'] = 32
        self.dataset['interpolation'] = 'bilinear'
        self.dataset['num_classes'] = 10

config = MergedConfig()

# Create preprocessor
data_aux_preprocessor = TestStandardPreProcessor(config)

# ID test dataset
id_dataset = ImglistDataset(
    name='cifar10_test',
    imglist_pth='data/benchmark_imglist/cifar10/test_cifar10.txt',
    data_dir='data/images_classic',
    num_classes=10,
    preprocessor=data_aux_preprocessor.transform,
    data_aux_preprocessor=data_aux_preprocessor.transform,
)

id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0)

# OOD datasets
ood_loaders = {}
for ood_name in ['cifar100', 'tin']:
    dataset = ImglistDataset(
        name=f'cifar10_ood_{ood_name}',
        imglist_pth=f'data/benchmark_imglist/cifar10/test_{ood_name}.txt',
        data_dir='data/images_classic',
        num_classes=10,
        preprocessor=data_aux_preprocessor.transform,
        data_aux_preprocessor=data_aux_preprocessor.transform,
    )
    ood_loaders[ood_name] = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# Model setup
def load_model(ckpt_path):
    model = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_energy(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def compute_auroc(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc * 100  # percentage

def evaluate_run(ckpt_path, run_name):
    print(f"\n=== Evaluating {run_name} ===")
    model = load_model(ckpt_path)
    
    # Get ID scores
    id_energy = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data']
            logits = model(data)
            energy = compute_energy(logits)
            id_energy.extend(energy.cpu().numpy().tolist())
    id_energy = np.array(id_energy)
    print(f"  ID samples: {len(id_energy)}")
    
    results = {}
    for ood_name, loader in ood_loaders.items():
        ood_energy = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data']
                logits = model(data)
                energy = compute_energy(logits)
                ood_energy.extend(energy.cpu().numpy().tolist())
        ood_energy = np.array(ood_energy)
        
        auroc = compute_auroc(id_energy, ood_energy)
        results[ood_name] = auroc
        print(f"  {ood_name}: {len(ood_energy)} samples, AUROC = {auroc:.2f}%")
    
    return results

# Run all three seeds
seeds = ['s0', 's1', 's2']
base_path = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'

all_results = {}
for seed in seeds:
    ckpt_path = os.path.join(base_path, seed, 'best.ckpt')
    all_results[seed] = evaluate_run(ckpt_path, seed)

# Compute aggregate
print("\n\n=== AGGREGATE RESULTS ===")
dataset_names = ['cifar100', 'tin']
run_metrics = {}
for seed in seeds:
    run_metrics[seed] = {d: all_results[seed][d] for d in dataset_names}

# Dataset mean per run, then mean across runs
run_means = []
for seed in seeds:
    run_mean = np.mean([all_results[seed][d] for d in dataset_names])
    run_means.append(run_mean)
    print(f"{seed}: dataset_mean = {run_mean:.4f}")

actual = np.mean(run_means)
print(f"\nFinal AUROC (dataset_mean_then_run_mean): {actual:.4f}%")

# Print the evidence line
evidence = {
    "metric": "near_ood_auroc",
    "actual": float(actual),
    "datasets": {
        "cifar100": len(ood_loaders['cifar100'].dataset),
        "tin": len(ood_loaders['tin'].dataset),
    },
    "run_metrics": run_metrics,
    "aggregation": "dataset_mean_then_run_mean"
}
print(f"\nREPRO_RESULT {json.dumps(evidence)}")
PYEOF
python eval_ebo.py
cat openood/utils/config.py
cat openood/utils/config.py | head -60
grep -n "class Config" openood/utils/config.py
sed -n '88,200p' openood/utils/config.py
cat > eval_ebo.py <<'PYEOF'
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

sys.path.insert(0, '.')

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

import torchvision.transforms as tvs_trans

# Normalization for CIFAR-10 (from openood/preprocessors/transform.py)
norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2470, 0.2435, 0.2616]

# Preprocessing: same as TestStandardPreProcessor
transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=norm_mean, std=norm_std),
])

# ID test dataset
id_dataset = ImglistDataset(
    name='cifar10_test',
    imglist_pth='data/benchmark_imglist/cifar10/test_cifar10.txt',
    data_dir='data/images_classic',
    num_classes=10,
    preprocessor=transform,
    data_aux_preprocessor=transform,
)

id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0)

# OOD datasets
ood_loaders = {}
for ood_name in ['cifar100', 'tin']:
    dataset = ImglistDataset(
        name=f'cifar10_ood_{ood_name}',
        imglist_pth=f'data/benchmark_imglist/cifar10/test_{ood_name}.txt',
        data_dir='data/images_classic',
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    ood_loaders[ood_name] = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# Model setup
def load_model(ckpt_path):
    model = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_energy(logits, temperature=1.0):
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def compute_auroc(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc * 100  # percentage

def evaluate_run(ckpt_path, run_name):
    print(f"\n=== Evaluating {run_name} ===")
    model = load_model(ckpt_path)
    
    # Get ID scores
    id_energy = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data']
            logits = model(data)
            energy = compute_energy(logits)
            id_energy.extend(energy.cpu().numpy().tolist())
    id_energy = np.array(id_energy)
    print(f"  ID samples: {len(id_energy)}")
    
    results = {}
    for ood_name, loader in ood_loaders.items():
        ood_energy = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data']
                logits = model(data)
                energy = compute_energy(logits)
                ood_energy.extend(energy.cpu().numpy().tolist())
        ood_energy = np.array(ood_energy)
        
        auroc = compute_auroc(id_energy, ood_energy)
        results[ood_name] = auroc
        print(f"  {ood_name}: {len(ood_energy)} samples, AUROC = {auroc:.2f}%")
    
    return results

# Run all three seeds
seeds = ['s0', 's1', 's2']
base_path = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'

all_results = {}
for seed in seeds:
    ckpt_path = os.path.join(base_path, seed, 'best.ckpt')
    all_results[seed] = evaluate_run(ckpt_path, seed)

# Compute aggregate
print("\n\n=== AGGREGATE RESULTS ===")
dataset_names = ['cifar100', 'tin']
run_metrics = {}
for seed in seeds:
    run_metrics[seed] = {d: all_results[seed][d] for d in dataset_names}

# Dataset mean per run, then mean across runs
run_means = []
for seed in seeds:
    run_mean = np.mean([all_results[seed][d] for d in dataset_names])
    run_means.append(run_mean)
    print(f"{seed}: dataset_mean = {run_mean:.4f}")

actual = np.mean(run_means)
print(f"\nFinal AUROC (dataset_mean_then_run_mean): {actual:.4f}%")

# Print the evidence line
evidence = {
    "metric": "near_ood_auroc",
    "actual": float(actual),
    "datasets": {
        "cifar100": len(ood_loaders['cifar100'].dataset),
        "tin": len(ood_loaders['tin'].dataset),
    },
    "run_metrics": run_metrics,
    "aggregation": "dataset_mean_then_run_mean"
}
print(f"\nREPRO_RESULT {json.dumps(evidence)}")
PYEOF
python eval_ebo.py
grep -r "ebo" --include="*.md" --include="*.txt" --include="*.yml" --include="*.yaml" . 2>/dev/null | head -20
find . -name "*.md" -exec grep -l -i "auroc\|near-ood\|near_ood" {} \;
grep -A 20 -i "cifar10\|near.ood\|auroc" README.md | head -60
grep -i "leaderboard\|benchmark\|result" README.md | head -20
cat openood/evaluation_api/evaluator.py
cat openood/evaluation_api/postprocessor.py
cat > eval_ebo_official.py <<'PYEOF'
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.evaluation_api import Evaluator

# Load model
def load_model(ckpt_path):
    model = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Run all three seeds
seeds = ['s0', 's1', 's2']
base_path = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'

all_results = {}
for seed in seeds:
    ckpt_path = os.path.join(base_path, seed, 'best.ckpt')
    net = load_model(ckpt_path)
    
    print(f"\n=== Evaluating {seed} ===")
    evaluator = Evaluator(
        net=net,
        id_name='cifar10',
        data_root='./data',
        config_root='./configs',
        postprocessor_name='ebo',
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    
    # Evaluate near-ood
    scores = evaluator.eval_ood(id_split='test')
    
    # Extract near-ood results
    # The evaluator returns a dict with ood results
    print(f"Results keys: {scores.keys() if isinstance(scores, dict) else type(scores)}")
    
    all_results[seed] = scores

print("\n\nAll results collected.")
PYEOF
python eval_ebo_official.py 2>&1 | head -50
cat openood/postprocessors/ebo_postprocessor.py
cat openood/evaluators/ood_evaluator.py
cat openood/evaluators/metrics.py
grep -i "ebo\|energy" README.md | head -20
find . -name "*.csv" -o -name "*.json" -o -name "*.txt" 2>/dev/null | grep -i "result\|benchmark\|leaderboard" | head -10
cat results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/log.txt
grep -r "87\.\|88\.\|89\.\|90\." --include="*.md" --include="*.txt" --include="*.csv" . 2>/dev/null | grep -i "auroc\|ebo" | head -10
