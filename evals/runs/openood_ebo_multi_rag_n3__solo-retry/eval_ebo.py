#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32."""

import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official test transform (from TestStandardPreProcessor for CIFAR-10)
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# Checkpoint paths (relative to root)
CHECKPOINTS = {
    's0': 'cifar10_resnet18_32x32_ce_s0/best.ckpt',
    's1': 'cifar10_resnet18_32x32_ce_s1/best.ckpt',
    's2': 'cifar10_resnet18_32x32_ce_s2/best.ckpt',
}

# Dataset configs (relative to root)
DATASETS = {
    'cifar100': {
        'imglist_pth': 'benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': 'data/cifar100',
    },
    'tin': {
        'imglist_pth': 'benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': 'data/tinyimagenet',
    },
}


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points. Higher OOD score -> more OOD-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

    # Sort by score descending (higher score = more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    if pos == 0 or neg == 0:
        return 50.0

    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg

    # Trapezoidal integration
    auroc = np.trapz(tpr, fpr)
    return auroc * 100.0


def energy_score(logits, temperature=1.0):
    """Energy-based OOD score (higher = more OOD)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing data, checkpoints, and benchmark_imglist')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # -----------------------------------------------------------------------
    # Build ID dataset (CIFAR-10 test)
    # -----------------------------------------------------------------------
    id_imglist = os.path.join(root, 'benchmark_imglist/cifar10/test.txt')
    id_data_dir = os.path.join(root, 'data/cifar10')

    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=10,
        preprocessor=TEST_TRANSFORM,
        data_aux_preprocessor=TEST_TRANSFORM,
    )
    id_loader = DataLoader(id_dataset, batch_size=64, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Build OOD datasets
    # -----------------------------------------------------------------------
    ood_loaders = {}
    for ds_name, ds_cfg in DATASETS.items():
        imglist_pth = os.path.join(root, ds_cfg['imglist_pth'])
        data_dir = os.path.join(root, ds_cfg['data_dir'])
        dataset = ImglistDataset(
            name=f'cifar10_test_{ds_name}',
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=10,
            preprocessor=TEST_TRANSFORM,
            data_aux_preprocessor=TEST_TRANSFORM,
        )
        ood_loaders[ds_name] = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Evaluate each checkpoint
    # -----------------------------------------------------------------------
    run_metrics = {}
    dataset_counts = {}

    for run_name, ckpt_rel in CHECKPOINTS.items():
        ckpt_path = os.path.join(root, ckpt_rel)
        if not os.path.isfile(ckpt_path):
            print(f'ERROR: checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif 'net' in state:
            model.load_state_dict(state['net'])
        else:
            model.load_state_dict(state)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = energy_score(logits).cpu().numpy()
                id_scores.append(scores)
        id_scores = np.concatenate(id_scores)

        run_metrics[run_name] = {}
        for ds_name, ood_loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch['data'].to(device)
                    logits = model(images)
                    scores = energy_score(logits).cpu().numpy()
                    ood_scores.append(scores)
            ood_scores = np.concatenate(ood_scores)

            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ds_name] = round(auroc, 2)
            dataset_counts[ds_name] = len(ood_scores)

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    run_avgs = []
    for run_name in ['s0', 's1', 's2']:
        ds_values = [run_metrics[run_name][ds] for ds in ['cifar100', 'tin']]
        run_avgs.append(np.mean(ds_values))
    actual = float(np.mean(run_avgs))

    # -----------------------------------------------------------------------
    # Print result
    # -----------------------------------------------------------------------
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 2),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
