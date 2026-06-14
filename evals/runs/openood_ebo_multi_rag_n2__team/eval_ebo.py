#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO evaluation for OpenOOD ResNet18_32x32 on CIFAR-10.

Reproduces the official OpenOOD EBO AUROC for CIFAR-10 near-OOD (CIFAR-100,
TinyImageNet) using the three seeds s0, s1, s2.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.transform import Convert, normalization_dict


def get_test_transform():
    """Build the CIFAR-10 test transform from openood/preprocessors/test_preprocessor.py
    and openood/preprocessors/transform.py.

    The pipeline is: Convert('RGB') -> Resize(32) -> CenterCrop(32) -> ToTensor -> Normalize
    """
    mean = normalization_dict['cifar10'][0]
    std = normalization_dict['cifar10'][1]
    transform = tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])
    return transform


def compute_ebo_auroc(id_scores, ood_scores):
    """Compute AUROC (percentage) from ID and OOD energy scores.

    EBO: lower energy -> more ID-like. We treat ID as positive class.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    # Sort by score descending (higher score = more OOD-like for energy)
    # Energy: lower is more ID, so we negate for standard ROC convention
    # Actually we want: ID positive, OOD negative. Lower energy -> ID.
    # We'll use -energy as score so higher = more ID-like.
    neg_scores = -scores
    sorted_indices = np.argsort(neg_scores)[::-1]
    sorted_labels = labels[sorted_indices]

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 0.0

    tpr = 0.0
    fpr = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auroc = 0.0

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tpr += 1.0 / pos_count
        else:
            fpr += 1.0 / neg_count
            auroc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr
            prev_tpr = tpr

    # Add the last segment
    auroc += tpr * (1.0 - prev_fpr)

    return auroc * 100.0  # Return as percentage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to results directory containing s0, s1, s2')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Check for subfolders
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if not subfolders:
        raise ValueError(f'No subfolders (s0, s1, ...) found in {root}')

    # Dataset paths (relative to data root)
    # We assume the standard OpenOOD benchmark structure
    data_root = './data/images_classic/'
    imglist_root = './data/benchmark_imglist/cifar10/'

    # ID dataset: CIFAR-10 test
    id_imglist_path = os.path.join(imglist_root, 'test.txt')
    # OOD datasets
    ood_datasets = {
        'cifar100': os.path.join(imglist_root, 'cifar100.txt'),
        'tin': os.path.join(imglist_root, 'tin.txt'),
    }

    transform = get_test_transform()

    # Load ID dataset once
    id_dataset = ImglistDataset(
        imglist_path=id_imglist_path,
        data_root=data_root,
        transform=transform,
    )
    id_loader = DataLoader(
        id_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Load OOD datasets once each
    ood_loaders = {}
    ood_sizes = {}
    for name, imglist_path in ood_datasets.items():
        dataset = ImglistDataset(
            imglist_path=imglist_path,
            data_root=data_root,
            transform=transform,
        )
        ood_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        ood_sizes[name] = len(dataset)

    # Per-run metrics
    run_metrics = {}
    all_run_aurocs = {name: [] for name in ood_datasets}

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')

        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}, skipping {run_name}', file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle potential 'net.' prefix in state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                logits = model(data)
                # EBO energy: -T * log(sum(exp(logits/T))), T=1
                energy = -torch.logsumexp(logits, dim=1)
                id_scores.append(energy.cpu().numpy())
        id_scores = np.concatenate(id_scores)

        # Compute OOD scores for each OOD dataset
        run_aurocs = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data']
                    logits = model(data)
                    energy = -torch.logsumexp(logits, dim=1)
                    ood_scores.append(energy.cpu().numpy())
            ood_scores = np.concatenate(ood_scores)

            auroc = compute_ebo_auroc(id_scores, ood_scores)
            run_aurocs[ood_name] = auroc
            all_run_aurocs[ood_name].append(auroc)

        run_metrics[run_name] = run_aurocs

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, for each run compute mean across datasets
    run_means = []
    for run_name, aurocs in run_metrics.items():
        run_mean = np.mean(list(aurocs.values()))
        run_means.append(run_mean)

    # Then mean of run means
    actual = np.mean(run_means)

    # Build result
    # datasets values are evaluated sample counts (ID + OOD)
    id_count = len(id_dataset)
    ood_counts = {name: len(ood_loaders[name].dataset) for name in ood_datasets}

    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': id_count + ood_counts['cifar100'],
            'tin': id_count + ood_counts['tin'],
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
