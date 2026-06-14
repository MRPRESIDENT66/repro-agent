#!/usr/bin/env python3
"""
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32
and official s0/s1/s2 checkpoints. CPU-only, offline.
"""
import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

# Direct imports from openood (no evaluators/postprocessors/evaluation_api)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# 1.  Test transform (exact copy of TestStandardPreProcessor pipeline)
# ---------------------------------------------------------------------------
# CIFAR-10 normalization from openood/preprocessors/transform.py
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# For CIFAR-10 the configs/datasets/cifar10/cifar10.yml sets:
#   pre_size: 32
#   image_size: 32
#   interpolation: bilinear
# We hardcode these values (they are the same for all CIFAR-10 test splits).
test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# 2.  EBO score function (energy-based)
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """EBO score = -logsumexp(logits / temperature)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# 3.  AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending (higher energy = more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0
    # TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)

# ---------------------------------------------------------------------------
# 4.  Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model: torch.nn.Module,
                        id_loader: DataLoader,
                        ood_loaders: dict,
                        device: torch.device) -> dict:
    """Return dict {ood_name: auroc_percent}."""
    model.eval()
    id_scores_list = []
    with torch.no_grad():
        for images, _ in id_loader:
            images = images.to(device)
            logits = model(images)
            scores = energy_score(logits).cpu().numpy()
            id_scores_list.append(scores)
    id_scores = np.concatenate(id_scores_list)

    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_scores_list = []
        with torch.no_grad():
            for images, _ in ood_loader:
                images = images.to(device)
                logits = model(images)
                scores = energy_score(logits).cpu().numpy()
                ood_scores_list.append(scores)
        ood_scores = np.concatenate(ood_scores_list)
        results[ood_name] = compute_auroc(id_scores, ood_scores)
    return results

# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing results/ and data/')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # -----------------------------------------------------------------------
    # Paths (following OpenOOD convention)
    # -----------------------------------------------------------------------
    # Checkpoints: results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt
    ckpt_dir = os.path.join(root, 'results',
                            'cifar10_resnet18_32x32_base_e100_lr0.1_default')
    # Image lists: data/cifar10/... and data/ood/...
    data_dir = os.path.join(root, 'data')

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    # ID: CIFAR-10 test
    id_list_path = os.path.join(data_dir, 'cifar10', 'test.txt')
    id_dataset = ImglistDataset(list_path=id_list_path, transform=test_transform)
    id_loader = DataLoader(id_dataset, batch_size=64, shuffle=False, num_workers=0)

    # OOD: CIFAR-100 and TinyImageNet (Near-OOD)
    ood_configs = {
        'cifar100': os.path.join(data_dir, 'cifar100', 'test.txt'),
        'tin': os.path.join(data_dir, 'ood', 'tiny_imagenet', 'test.txt'),
    }
    ood_loaders = {}
    for name, list_path in ood_configs.items():
        dataset = ImglistDataset(list_path=list_path, transform=test_transform)
        ood_loaders[name] = DataLoader(dataset, batch_size=64, shuffle=False,
                                       num_workers=0)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = ResNet18_32x32(num_classes=10)
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Evaluate each checkpoint
    # -----------------------------------------------------------------------
    run_keys = ['s0', 's1', 's2']
    run_metrics = {}
    dataset_counts = {}

    for run_key in run_keys:
        ckpt_path = os.path.join(ckpt_dir, run_key, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        state = torch.load(ckpt_path, map_location=device)
        # The checkpoint may contain 'state_dict' or be the state_dict itself
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)

        results = evaluate_checkpoint(model, id_loader, ood_loaders, device)
        run_metrics[run_key] = results
        # Count samples (first batch to get dataset size)
        if not dataset_counts:
            dataset_counts['cifar100'] = len(ood_loaders['cifar100'].dataset)
            dataset_counts['tin'] = len(ood_loaders['tin'].dataset)

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # For each run, compute mean over datasets
    run_means = []
    for run_key in run_keys:
        vals = list(run_metrics[run_key].values())
        run_mean = float(np.mean(vals))
        run_means.append(run_mean)
    actual = float(np.mean(run_means))

    # -----------------------------------------------------------------------
    # Print the required JSON line
    # -----------------------------------------------------------------------
    result = {
        "metric": "near_ood_auroc",
        "actual": actual,
        "datasets": dataset_counts,
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean"
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == '__main__':
    main()
