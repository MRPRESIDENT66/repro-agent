#!/usr/bin/env python3
"""
eval_ebo.py - CPU-safe EBO evaluation for CIFAR-10 near-OOD detection.

Reproduces the OpenOOD EBO evaluation using official ResNet18_32x32 checkpoints
and benchmark image lists. Computes AUROC for CIFAR-100 and TinyImageNet (near-OOD)
across three seeds (s0, s1, s2) and reports aggregated results.

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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Only import the specific network class, not the broad openood package
from openood.networks.resnet18_32x32 import ResNet18_32x32


# ---------------------------------------------------------------------------
# Dataset for benchmark image lists
# ---------------------------------------------------------------------------

class ImglistDataset(Dataset):
    """Dataset from OpenOOD benchmark imglist files."""
    def __init__(self, imglist_path, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        with open(imglist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = os.path.join(data_dir, parts[0])
                    label = int(parts[1])
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# CIFAR-10 preprocessing (same as OpenOOD base_preprocessor)
# ---------------------------------------------------------------------------

def get_cifar10_preprocessor():
    """Return the standard CIFAR-10 test preprocessor."""
    return transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])


# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------

def compute_ebo_scores(net, dataloader, temperature=1.0):
    """Compute EBO energy scores for all samples in dataloader.
    
    EBO energy: E(x) = -T * logsumexp(f(x)/T)
    Higher energy = more OOD-like.
    """
    net.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = net(inputs)
            # EBO energy: -T * logsumexp(f(x)/T)
            energy = -temperature * torch.logsumexp(outputs / temperature, dim=1)
            all_scores.append(energy.cpu())
            all_labels.append(labels)
    return torch.cat(all_scores), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC (Area Under the Receiver Operating Characteristic curve).
    
    Higher energy = more OOD-like. Returns percentage (0-100).
    """
    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([torch.zeros(len(id_scores)), torch.ones(len(ood_scores))])
    
    # Sort by score (descending: higher score = more OOD-like)
    sorted_scores, sort_idx = torch.sort(scores, descending=True)
    sorted_labels = labels[sort_idx]
    
    # Compute TPR and FPR
    pos_count = sorted_labels.sum().item()
    neg_count = len(sorted_labels) - pos_count
    
    if pos_count == 0 or neg_count == 0:
        return 50.0
    
    tpr = 0.0
    fpr = 0.0
    auroc = 0.0
    
    for i in range(len(sorted_scores)):
        if sorted_labels[i] == 1:
            tpr += 1.0 / pos_count
        else:
            auroc += tpr * (1.0 / neg_count)
            fpr += 1.0 / neg_count
    
    return auroc * 100.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_seed(net, seed_dir, data_root, batch_size=200):
    """Evaluate a single seed checkpoint."""
    checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state_dict)
    
    # Setup preprocessing
    transform = get_cifar10_preprocessor()
    
    # ID dataset (CIFAR-10 test)
    id_dataset = ImglistDataset(
        imglist_path=os.path.join(data_root, 'benchmark_imglist/cifar10/test_cifar10.txt'),
        data_dir=os.path.join(data_root, 'images_classic/'),
        transform=transform
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # OOD datasets (near-OOD: CIFAR-100 and TinyImageNet)
    ood_datasets = {
        'cifar100': ImglistDataset(
            imglist_path=os.path.join(data_root, 'benchmark_imglist/cifar10/test_cifar100.txt'),
            data_dir=os.path.join(data_root, 'images_classic/'),
            transform=transform
        ),
        'tin': ImglistDataset(
            imglist_path=os.path.join(data_root, 'benchmark_imglist/cifar10/test_tin.txt'),
            data_dir=os.path.join(data_root, 'images_classic/'),
            transform=transform
        )
    }
    
    # Compute ID scores
    id_scores, _ = compute_ebo_scores(net, id_loader)
    
    # Compute OOD scores and AUROC for each dataset
    results = {}
    for name, dataset in ood_datasets.items():
        ood_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        ood_scores, _ = compute_ebo_scores(net, ood_loader)
        auroc = compute_auroc(id_scores, ood_scores)
        results[name] = auroc
    
    return results


def main():
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC for CIFAR-10')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size for DataLoader')
    args = parser.parse_args()
    
    root = args.root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If script is in project root, data is at ./data
    # If script is in a subdirectory, data is at ../data
    if os.path.exists(os.path.join(script_dir, 'data', 'benchmark_imglist')):
        data_root = os.path.join(script_dir, 'data')
    else:
        data_root = os.path.join(os.path.dirname(script_dir), 'data')
    
    # Find seed subfolders
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if len(seed_dirs) == 0:
        print(f'ERROR: No seed subfolders found in {root}', file=sys.stderr)
        sys.exit(1)
    
    # Initialize network using OpenOOD's ResNet18_32x32
    net = ResNet18_32x32(num_classes=10)
    
    # Evaluate each seed
    all_run_metrics = {}
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        print(f'Evaluating {seed_name}...', file=sys.stderr)
        try:
            results = evaluate_seed(net, seed_dir, data_root, args.batch_size)
            all_run_metrics[seed_name] = results
            print(f'  {seed_name}: cifar100={results["cifar100"]:.2f}, tin={results["tin"]:.2f}', file=sys.stderr)
        except Exception as e:
            print(f'  {seed_name}: ERROR - {e}', file=sys.stderr)
            sys.exit(1)
    
    # Compute aggregation: dataset mean within each run, then mean of runs
    dataset_means = {}
    for dataset_name in ['cifar100', 'tin']:
        values = [all_run_metrics[seed][dataset_name] for seed in all_run_metrics]
        dataset_means[dataset_name] = np.mean(values)
    
    run_means = []
    for seed in all_run_metrics:
        run_mean = np.mean([all_run_metrics[seed][d] for d in ['cifar100', 'tin']])
        run_means.append(run_mean)
    
    actual = np.mean(run_means)
    
    # Build output dictionary with proper types
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': len(all_run_metrics),
            'tin': len(all_run_metrics)
        },
        'run_metrics': {
            seed: {
                'cifar100': float(all_run_metrics[seed]['cifar100']),
                'tin': float(all_run_metrics[seed]['tin'])
            }
            for seed in all_run_metrics
        },
        'aggregation': 'dataset_mean_then_run_mean'
    }
    
    # Print the required output line - this is the ONLY stdout output
    print(f'REPRO_RESULT {json.dumps(output)}')
    
    # All other output goes to stderr
    print(f'\nSummary:', file=sys.stderr)
    print(f'  Dataset means: cifar100={dataset_means["cifar100"]:.2f}, tin={dataset_means["tin"]:.2f}', file=sys.stderr)
    print(f'  Run means: {[f"{m:.2f}" for m in run_means]}', file=sys.stderr)
    print(f'  Final AUROC: {actual:.2f}', file=sys.stderr)
    sys.exit(0)


if __name__ == '__main__':
    main()
