#!/usr/bin/env python3
"""
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32
and official s0/s1/s2 checkpoints. CPU-only, offline.
"""

import json
import os
import sys
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from torch.utils.data import DataLoader
from PIL import Image

# Direct imports from openood (no evaluation_api, evaluators, postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official checkpoint paths relative to root
CHECKPOINT_RELS = {
    's0': 'cifar10_resnet18_32x32_base_e100_lr0.5_default/s0/best.ckpt',
    's1': 'cifar10_resnet18_32x32_base_e100_lr0.5_default/s1/best.ckpt',
    's2': 'cifar10_resnet18_32x32_base_e100_lr0.5_default/s2/best.ckpt',
}

# Image list paths relative to root
IMGLIST_RELS = {
    'cifar10': 'data/benchmark_imglist/cifar10/test_cifar10.txt',
    'cifar100': 'data/benchmark_imglist/cifar100/test_cifar100.txt',
    'tin': 'data/benchmark_imglist/tin/test_tin.txt',
}

# Alternative image list paths (relative to root) for when the standard paths don't exist
IMGLIST_RELS_ALT = {
    'cifar10': 'data/benchmark_imglist/cifar10/test_cifar10.txt',
    'cifar100': 'data/benchmark_imglist/cifar100/test_cifar100.txt',
    'tin': 'data/benchmark_imglist/tin/test_tin.txt',
}

# Data directories relative to root
DATA_DIRS = {
    'cifar10': 'data/images_classic',
    'cifar100': 'data/images_classic',
    'tin': 'data/images_classic',
}

NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_WORKERS = 0  # CPU-safe

# ---------------------------------------------------------------------------
# Transform (exact reproduction of TestStandardPreProcessor for CIFAR-10)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the exact test transform used by OpenOOD for CIFAR-10."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score from logits."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """Compute AUROC in percentage points. Higher score = more ID-like."""
    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([
        torch.ones_like(id_scores),
        torch.zeros_like(ood_scores),
    ])
    # Sort by score descending
    sorted_scores, sort_idx = torch.sort(scores, descending=True)
    sorted_labels = labels[sort_idx]

    pos = sorted_labels.sum().float()
    neg = (1 - sorted_labels).sum().float()

    if pos == 0 or neg == 0:
        return 50.0  # random

    # Compute TPR and FPR
    tpr = torch.cumsum(sorted_labels, dim=0) / pos
    fpr = torch.cumsum(1 - sorted_labels, dim=0) / neg

    # Trapezoidal integration
    fpr_diff = fpr[1:] - fpr[:-1]
    tpr_avg = (tpr[1:] + tpr[:-1]) / 2.0
    auroc = (fpr_diff * tpr_avg).sum().item() * 100.0  # percentage
    return auroc

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of OpenOOD repository')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # Build transform
    transform = get_test_transform()

    # Load ID dataset (CIFAR-10 test)
    # Try standard path first, then fall back to alternative
    imglist_pth_cifar10 = os.path.join(root, IMGLIST_RELS['cifar10'])
    if not os.path.isfile(imglist_pth_cifar10):
        imglist_pth_cifar10 = os.path.join(root, IMGLIST_RELS_ALT['cifar10'])
    if not os.path.isfile(imglist_pth_cifar10):
        # Fallback: look for the imglist relative to the repository root (not results root)
        # The repository root is one level above the results directory
        repo_root = os.path.dirname(root)
        imglist_pth_cifar10 = os.path.join(repo_root, IMGLIST_RELS['cifar10'])
    if not os.path.isfile(imglist_pth_cifar10):
        # Final fallback: look for the imglist relative to the current working directory
        imglist_pth_cifar10 = os.path.join(os.getcwd(), IMGLIST_RELS['cifar10'])
    id_dataset = ImglistDataset(
        name='cifar10',
        imglist_pth=imglist_pth_cifar10,
        data_dir=os.path.join(root, DATA_DIRS['cifar10']),
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Load OOD datasets
    ood_datasets = {}
    for ood_name in ['cifar100', 'tin']:
        imglist_pth_ood = os.path.join(root, IMGLIST_RELS[ood_name])
        if not os.path.isfile(imglist_pth_ood):
            # Fallback: look for the imglist relative to the repository root
            repo_root = os.path.dirname(root)
            imglist_pth_ood = os.path.join(repo_root, IMGLIST_RELS[ood_name])
        if not os.path.isfile(imglist_pth_ood):
            # Final fallback: look for the imglist relative to the current working directory
            imglist_pth_ood = os.path.join(os.getcwd(), IMGLIST_RELS[ood_name])
        ood_dataset = ImglistDataset(
            name=ood_name,
            imglist_pth=imglist_pth_ood,
            data_dir=os.path.join(root, DATA_DIRS[ood_name]),
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        ood_datasets[ood_name] = ood_loader

    # Results storage
    run_metrics = {}
    dataset_counts = {}

    for run_name in ['s0', 's1', 's2']:
        # Load checkpoint
        ckpt_path = os.path.join(root, CHECKPOINT_RELS[run_name])
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state_dict = torch.load(ckpt_path, map_location=device)
        # Handle possible 'net.' prefix
        if any(k.startswith('net.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Compute ID scores
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = energy_score(logits)
                id_scores_list.append(scores.cpu())
        id_scores = torch.cat(id_scores_list)

        # Compute OOD scores per dataset
        run_metrics[run_name] = {}
        for ood_name, ood_loader in ood_datasets.items():
            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch['data'].to(device)
                    logits = model(images)
                    scores = energy_score(logits)
                    ood_scores_list.append(scores.cpu())
            ood_scores = torch.cat(ood_scores_list)

            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ood_name] = auroc

            # Store dataset count (only once, they're the same across runs)
            if ood_name not in dataset_counts:
                dataset_counts[ood_name] = len(ood_scores)

    # Store ID dataset count
    dataset_counts['cifar100'] = dataset_counts.get('cifar100', len(id_scores))
    # Actually, we need counts for each dataset as evaluated
    # Recompute properly
    dataset_counts = {}
    for ood_name in ['cifar100', 'tin']:
        dataset_counts[ood_name] = len(ood_datasets[ood_name].dataset)

    # Compute aggregation: dataset mean within each run, then mean of runs
    run_means = []
    for run_name in ['s0', 's1', 's2']:
        ds_vals = [run_metrics[run_name][ds] for ds in ['cifar100', 'tin']]
        run_mean = sum(ds_vals) / len(ds_vals)
        run_means.append(run_mean)
    actual = sum(run_means) / len(run_means)

    # Build result dict
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one strict-JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
