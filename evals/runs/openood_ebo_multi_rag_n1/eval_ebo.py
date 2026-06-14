#!/usr/bin/env python3
"""eval_ebo.py — CPU-safe EBO Near-OOD AUROC for CIFAR-10.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants (from openood/preprocessors/transform.py)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Complete test transform pipeline from TestStandardPreProcessor
# (openood/preprocessors/test_preprocessor.py)
# Includes: Convert('RGB'), Resize, CenterCrop, ToTensor, Normalize
test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Dataset paths (standard OpenOOD benchmark locations)
# ---------------------------------------------------------------------------
# ID test: CIFAR-10 test set
ID_IMG_LIST = './data/benchmark_imglist/cifar10/test_cifar10.txt'
ID_DATA_ROOT = './data/images_classic'

# Near-OOD datasets
OOD_CONFIGS = {
    'cifar100': {
        'imglist': './data/benchmark_imglist/cifar10/test_cifar100.txt',
        'data_root': './data/images_classic',
    },
    'tin': {
        'imglist': './data/benchmark_imglist/cifar10/test_tin.txt',
        'data_root': './data/images_classic',
    },
}

# ---------------------------------------------------------------------------
# EBO score function (from EBOPostprocessor)
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute EBO confidence score.

    Higher score = more confident (ID-like).
    """
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation (standard binary classification)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points.

    ID is positive class, OOD is negative class.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    # True positive rate and false positive rate
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 50.0  # Random performance

    tpr = np.cumsum(sorted_labels == 1) / pos_count
    fpr = np.cumsum(sorted_labels == 0) / neg_count

    # Trapezoidal integration
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)  # Convert to percentage

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate(root: str):
    """Run EBO evaluation for all seeds and OOD datasets."""
    # Discover seed subfolders
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        raise ValueError(f'No seed subfolders (s*) found in {root}')

    # Prepare datasets
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=ID_IMG_LIST,
        data_dir=ID_DATA_ROOT,
        num_classes=10,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0)

    ood_datasets = {}
    ood_loaders = {}
    for name, cfg in OOD_CONFIGS.items():
        ds = ImglistDataset(
            name=f'{name}_test',
            imglist_pth=cfg['imglist'],
            data_dir=cfg['data_root'],
            num_classes=10,
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform,
        )
        ood_datasets[name] = ds
        ood_loaders[name] = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # Store per-seed, per-dataset AUROC
    run_metrics = {}
    all_dataset_aurocs = {name: [] for name in OOD_CONFIGS}

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}, skipping {seed_name}')
            continue

        # Load model
        net = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle possible 'net.' prefix
        if any(k.startswith('net.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        net.load_state_dict(state_dict)
        net.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']  # Already tensor from transform
                logits = net(data)
                scores = ebo_score(logits, temperature=1.0)
                id_scores.append(scores.numpy())
        id_scores = np.concatenate(id_scores)

        # Compute OOD scores for each dataset
        seed_aurocs = {}
        for name, loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in loader:
                    data = batch['data']
                    logits = net(data)
                    scores = ebo_score(logits, temperature=1.0)
                    ood_scores.append(scores.numpy())
            ood_scores = np.concatenate(ood_scores)

            auroc = compute_auroc(id_scores, ood_scores)
            seed_aurocs[name] = auroc
            all_dataset_aurocs[name].append(auroc)

        run_metrics[seed_name] = seed_aurocs

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-run dataset mean
    per_run_dataset_means = []
    for seed_name, seed_aurocs in run_metrics.items():
        dataset_mean = np.mean(list(seed_aurocs.values()))
        per_run_dataset_means.append(dataset_mean)

    # Then mean of runs
    actual = float(np.mean(per_run_dataset_means))

    # Compute dataset means across runs for reporting
    dataset_means = {name: float(np.mean(aurocs)) for name, aurocs in all_dataset_aurocs.items()}
    dataset_counts = {name: len(aurocs) for name, aurocs in all_dataset_aurocs.items()}

    # Build result with proper structure
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': {
            's0': run_metrics.get('s0', {}),
            's1': run_metrics.get('s1', {}),
            's2': run_metrics.get('s2', {}),
        },
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC for CIFAR-10')
    parser.add_argument('--root', type=str, required=True,
                        help='Path to results directory containing s0, s1, s2 subfolders')
    args = parser.parse_args()
    evaluate(args.root)
