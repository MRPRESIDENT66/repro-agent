#!/usr/bin/env python3
"""
Reproduce official OpenOOD EBO Near-OOD AUROC for CIFAR-10.
CPU-only, offline, using official checkpoints and dataset lists.
"""

import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Import only the required modules – no evaluation_api, evaluators, or postprocessors.
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 256
NUM_WORKERS = 0  # CPU-only, avoid multiprocessing issues

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official test transform for CIFAR-10 (from TestStandardPreProcessor)
# Resize to 32, CenterCrop to 32, ToTensor, Normalize
test_transform = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# EBO scoring function
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score: -T * logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation (percentage)
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
        return 50.0  # random

    # TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg

    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)  # percentage

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing data/ and checkpoints/')
    args = parser.parse_args()

    root = args.root
    data_dir = os.path.join(root, 'data')
    ckpt_dir = os.path.join(root, 'checkpoints')

    # -----------------------------------------------------------------------
    # Dataset paths (official OpenOOD structure)
    # -----------------------------------------------------------------------
    # ID: CIFAR-10 test
    id_imglist = os.path.join(data_dir, 'cifar10', 'test.txt')
    id_data_dir = os.path.join(data_dir, 'cifar10', 'data')

    # Near-OOD: CIFAR-100
    ood1_imglist = os.path.join(data_dir, 'cifar100', 'test.txt')
    ood1_data_dir = os.path.join(data_dir, 'cifar100', 'data')

    # Near-OOD: TinyImageNet (crop)
    ood2_imglist = os.path.join(data_dir, 'tinyimagenet_crop', 'test.txt')
    ood2_data_dir = os.path.join(data_dir, 'tinyimagenet_crop', 'data')

    # -----------------------------------------------------------------------
    # Checkpoint paths (official s0, s1, s2)
    # -----------------------------------------------------------------------
    ckpt_files = {
        's0': os.path.join(ckpt_dir, 'resnet18_32x32_ce_s0.ckpt'),
        's1': os.path.join(ckpt_dir, 'resnet18_32x32_ce_s1.ckpt'),
        's2': os.path.join(ckpt_dir, 'resnet18_32x32_ce_s2.ckpt'),
    }

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )

    ood1_dataset = ImglistDataset(
        name='cifar100_test',
        imglist_pth=ood1_imglist,
        data_dir=ood1_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )

    ood2_dataset = ImglistDataset(
        name='tinyimagenet_test',
        imglist_pth=ood2_imglist,
        data_dir=ood2_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )

    # -----------------------------------------------------------------------
    # DataLoaders (CPU, batch_size=256)
    # -----------------------------------------------------------------------
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS)
    ood1_loader = DataLoader(ood1_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)
    ood2_loader = DataLoader(ood2_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    # -----------------------------------------------------------------------
    # Evaluate each checkpoint
    # -----------------------------------------------------------------------
    run_metrics = {}  # {run_name: {dataset_name: auroc}}

    for run_name, ckpt_path in ckpt_files.items():
        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location='cpu')
        # State dict may be wrapped in 'state_dict' key or directly
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()

        # Inference function
        @torch.no_grad()
        def get_energy_scores(loader):
            scores = []
            for batch in loader:
                images = batch['data']  # tensor
                logits = model(images)
                energy = energy_score(logits)
                scores.append(energy.cpu().numpy())
            return np.concatenate(scores)

        # ID scores
        id_scores = get_energy_scores(id_loader)

        # OOD1 (CIFAR-100)
        ood1_scores = get_energy_scores(ood1_loader)
        auroc_ood1 = compute_auroc(id_scores, ood1_scores)

        # OOD2 (TinyImageNet)
        ood2_scores = get_energy_scores(ood2_loader)
        auroc_ood2 = compute_auroc(id_scores, ood2_scores)

        run_metrics[run_name] = {
            'cifar100': auroc_ood1,
            'tin': auroc_ood2,
        }

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # Dataset means per run
    dataset_means = {}
    for run_name, metrics in run_metrics.items():
        dataset_means[run_name] = np.mean(list(metrics.values()))

    # Final actual = mean of run-level dataset means
    actual = float(np.mean(list(dataset_means.values())))

    # Dataset counts (number of evaluated samples)
    dataset_counts = {
        'cifar100': len(ood1_dataset),
        'tin': len(ood2_dataset),
    }

    # -----------------------------------------------------------------------
    # Print result (exact JSON line)
    # -----------------------------------------------------------------------
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
