#!/usr/bin/env python3
"""CPU-safe EBO evaluation for CIFAR-10 ResNet18_32x32 (s0/s1/s2).

Verification-driven reproduction script.  Uses only direct OpenOOD imports,
the official ImglistDataset, and a minimal local EBO/AUROC implementation.
Prints exactly one REPRO_RESULT JSON line on success.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader

# Direct model/dataset imports – no evaluation_api, evaluators, postprocessors.
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants from repository source (openood/preprocessors/transform.py)
# ---------------------------------------------------------------------------
NORMALIZATION_DICT = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
}

INTERPOLATION_MODES = {
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
}

NUM_CLASSES = 10
BATCH_SIZE = 200
TEMPERATURE = 1.0  # EBO default (scripts/ood/ebo/cifar10_test_ood_ebo.sh)

# ---------------------------------------------------------------------------
# Build the test transform (exactly as in openood/preprocessors/transform.py)
# ---------------------------------------------------------------------------
def build_test_transform():
    """Return the torchvision test transform for CIFAR-10."""
    mean, std = NORMALIZATION_DICT['cifar10']
    transform = tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=INTERPOLATION_MODES['bilinear']),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])
    return transform

# ---------------------------------------------------------------------------
# EBO score function (energy-based)
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Compute energy score: temperature * logsumexp(logits / temperature)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (percentage 0–100)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Return AUROC in percentage points (0–100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending (higher energy → more OOD-like)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels_sorted == 1)
    neg = np.sum(labels_sorted == 0)
    if pos == 0 or neg == 0:
        return 50.0
    # TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Parent dir containing s0/, s1/, s2/')
    args = parser.parse_args()

    root = args.root
    # Discover seed subfolders
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        raise ValueError(f'No seed subfolders found in {root}')

    # Build transform once
    transform = build_test_transform()

    # Dataset paths (relative to repo root, assumed to be current dir or parent)
    data_dir = './data/images_classic/'
    id_imglist = './data/benchmark_imglist/cifar10/test_cifar10.txt'
    ood_imglists = {
        'cifar100': './data/benchmark_imglist/cifar10/test_cifar100.txt',
        'tin': './data/benchmark_imglist/cifar10/test_tin.txt',
    }

    # Pre-create datasets (they are small, we can reuse them)
    # ID dataset
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,  # same transform for aux
    )
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # OOD datasets
    ood_loaders = {}
    for ood_name, imglist in ood_imglists.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=imglist,
            data_dir=data_dir,
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loaders[ood_name] = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Store per-seed, per-dataset AUROC
    run_metrics = {}  # {seed_label: {dataset: auroc}}
    dataset_counts = {}  # {dataset: count} – will be filled from first run

    device = torch.device('cpu')

    for seed_dir in seed_dirs:
        seed_label = os.path.basename(seed_dir)  # e.g., 's0'
        ckpt_path = os.path.join(seed_dir, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f'Warning: checkpoint not found at {ckpt_path}, skipping {seed_label}', file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location=device)
        # The checkpoint may contain 'state_dict' or be the state_dict itself
        if 'state_dict' in state:
            state = state['state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state)
        model.eval()
        model.to(device)

        # Compute ID scores once per seed
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data'].to(device)
                logits = model(data)
                scores = ebo_score(logits, TEMPERATURE)
                id_scores_list.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores_list)

        # Store dataset count from first run
        if not dataset_counts:
            dataset_counts['cifar100'] = len(ood_loaders['cifar100'].dataset)
            dataset_counts['tin'] = len(ood_loaders['tin'].dataset)

        # Per OOD dataset
        seed_metrics = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data'].to(device)
                    logits = model(data)
                    scores = ebo_score(logits, TEMPERATURE)
                    ood_scores_list.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores_list)
            auroc = compute_auroc(id_scores, ood_scores)
            seed_metrics[ood_name] = auroc

        run_metrics[seed_label] = seed_metrics

    # -----------------------------------------------------------------------
    # Aggregation: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # run_metrics: {s0: {cifar100: x, tin: y}, s1: ..., s2: ...}
    # For each run, compute mean over datasets, then average across runs.
    run_means = []
    for seed_label, metrics in run_metrics.items():
        run_mean = np.mean(list(metrics.values()))
        run_means.append(run_mean)
    actual = float(np.mean(run_means))

    # Build output JSON
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
