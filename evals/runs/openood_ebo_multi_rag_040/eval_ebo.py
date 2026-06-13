#!/usr/bin/env python3
"""eval_ebo.py — CPU-safe EBO Near-OOD AUROC for CIFAR-10.

Reproduces the official OpenOOD evaluation for ResNet18_32x32 on CIFAR-10
using EBO postprocessor. Prints exactly one REPRO_RESULT JSON line.
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset


# ---------------------------------------------------------------------------
# Transform pipeline (from openood/preprocessors/transform.py and test_preprocessor.py)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the standard CIFAR-10 test transform (no Convert needed for PIL)."""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    return transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute negative energy score: -E(x) = logsumexp(f(x)/T)."""
    return torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation (OOD as positive class)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC treating OOD as positive class.
    
    ID samples have higher energy scores than OOD samples, so we negate
    the scores before roc_auc_score (consistent with OpenOOD's metrics.py).
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # Negate because higher energy → more OOD-like
    return roc_auc_score(labels, -scores) * 100.0  # percentage


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root (e.g., ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default)')
    args = parser.parse_args()

    root = args.root
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Checkpoint subfolders
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if not subfolders:
        raise ValueError(f'No s* subfolders found in {root}')

    # Dataset definitions (from openood/evaluation_api/datasets.py)
    id_info = {
        'data_dir': 'images_classic/',
        'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
    }
    ood_datasets = {
        'cifar100': {
            'data_dir': 'images_classic/',
            'imglist_path': 'benchmark_imglist/cifar10/test_cifar100.txt'
        },
        'tin': {
            'data_dir': 'images_classic/',
            'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
        }
    }

    transform = get_test_transform()

    # Prepare ID dataset (shared across runs)
    id_dataset = ImglistDataset(
        name='cifar10',
        imglist_pth=os.path.join(data_root, id_info['imglist_path']),
        data_dir=os.path.join(data_root, id_info['data_dir']),
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=4)

    # Prepare OOD datasets
    ood_loaders = {}
    ood_datasets_objs = {}
    for name, info in ood_datasets.items():
        dataset = ImglistDataset(
            name=name,
            imglist_pth=os.path.join(data_root, info['imglist_path']),
            data_dir=os.path.join(data_root, info['data_dir']),
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform
        )
        ood_loaders[name] = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4)
        ood_datasets_objs[name] = dataset

    # Run evaluation for each seed
    run_metrics = {}
    for subfolder in subfolders:
        seed_name = os.path.basename(subfolder)
        ckpt_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f'Warning: checkpoint not found at {ckpt_path}, skipping')
            continue

        # Load model
        net = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state)
        net.eval()

        # Inference on ID
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                logits = net(data)
                scores = ebo_score(logits)
                id_scores.append(scores.numpy())
        id_scores = np.concatenate(id_scores)

        # Inference on OOD datasets
        ood_aurocs = {}
        for ood_name, loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in loader:
                    data = batch['data']
                    logits = net(data)
                    scores = ebo_score(logits)
                    ood_scores.append(scores.numpy())
            ood_scores = np.concatenate(ood_scores)
            ood_aurocs[ood_name] = compute_auroc(id_scores, ood_scores)

        run_metrics[seed_name] = ood_aurocs

    # Aggregate: dataset mean within each run, then mean of runs
    dataset_names = list(ood_datasets.keys())
    dataset_means = {d: [] for d in dataset_names}
    for seed_name, metrics in run_metrics.items():
        for d in dataset_names:
            dataset_means[d].append(metrics[d])

    # Dataset mean across runs
    dataset_avg = {d: np.mean(vals) for d, vals in dataset_means.items()}
    # Overall mean (mean of dataset means)
    actual = np.mean(list(dataset_avg.values()))

    # Build result with evaluated sample counts
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': len(ood_datasets_objs['cifar100']),
            'tin': len(ood_datasets_objs['tin'])
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
