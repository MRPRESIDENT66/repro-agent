#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC reproduction for CIFAR-10.

Reproduces the official OpenOOD EBO evaluation for ResNet18_32x32 on CIFAR-10
with near-OOD datasets CIFAR-100 and TinyImageNet. Uses only direct module
imports from openood.networks and openood.datasets, avoiding heavy
evaluation_api, evaluators, and postprocessors packages.

Usage:
    python eval_ebo.py --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms

# Direct imports from OpenOOD — no evaluation_api, evaluators, or postprocessors
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants from OpenOOD repository
# ---------------------------------------------------------------------------
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}

# CIFAR-10 normalization from openood/preprocessors/transform.py
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Near-OOD datasets for CIFAR-10 (from OpenOOD configs)
NEAR_OOD_DATASETS = ['cifar100', 'tin']

# Benchmark image list paths (relative to data_dir)
BENCHMARK_IMG_LISTS = {
    'cifar10': 'data/benchmark_imglist/cifar10/test_cifar10.txt',
    'cifar100': 'data/benchmark_imglist/cifar100/test_cifar100.txt',
    'tin': 'data/benchmark_imglist/cifar100/test_tin.txt',
}

# Data directories (relative to --root parent or absolute)
DATA_DIRS = {
    'cifar10': 'data/images_classic',
    'cifar100': 'data/images_classic',
    'tin': 'data/images_classic',
}

# ---------------------------------------------------------------------------
# Test transform (from openood/preprocessors/transform.py TestStandardPreProcessor)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Build the test transform for CIFAR-10 ResNet18_32x32."""
    return transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits, temperature=1.0):
    """Compute EBO (energy) scores from logits.
    
    Args:
        logits: Tensor of shape (N, num_classes)
        temperature: Temperature parameter (default 1.0)
    
    Returns:
        energy_scores: Tensor of shape (N,) — higher = more ID-like
    """
    # Energy = temperature * logsumexp(logits / temperature, dim=1)
    # For EBO, we negate so that higher scores = more ID
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (from openood/evaluators/metrics.py)
# ---------------------------------------------------------------------------
def compute_auroc(conf_scores, labels):
    """Compute AUROC following OpenOOD convention.
    
    Args:
        conf_scores: numpy array of confidence/energy scores (higher = more ID)
        labels: numpy array where -1 = OOD, else class label
    
    Returns:
        auroc: AUROC value in [0, 1] (not percentage)
    """
    ood_indicator = np.zeros_like(labels)
    ood_indicator[labels == -1] = 1
    
    # Negate conf scores because OpenOOD assumes ID samples have larger conf
    fpr_list, tpr_list, _ = metrics.roc_curve(ood_indicator, -conf_scores)
    return metrics.auc(fpr_list, tpr_list)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_name, data_root, transform):
    """Load a dataset using ImglistDataset.
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'tin'
        data_root: Root directory containing data/ and benchmark_imglist/
        transform: torchvision transform to apply
    
    Returns:
        ImglistDataset instance
    """
    imglist_pth = os.path.join(data_root, BENCHMARK_IMG_LISTS[dataset_name])
    data_dir = os.path.join(data_root, DATA_DIRS[dataset_name])
    
    # For OOD datasets, num_classes doesn't matter for evaluation
    num_classes = NUM_CLASSES.get(dataset_name, 10)
    
    return ImglistDataset(
        name=dataset_name,
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=num_classes,
        preprocessor=transform,
        data_aux_preprocessor=transform,  # Same transform for aux
    )

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='EBO Near-OOD AUROC reproduction for CIFAR-10'
    )
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Path to results directory containing s0/s1/s2 subfolders'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Batch size for DataLoader'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='EBO temperature parameter'
    )
    args = parser.parse_args()
    
    root = args.root
    batch_size = args.batch_size
    temperature = args.temperature
    
    # Validate root directory
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        raise ValueError(f'No seed subdirectories (s*) found in {root}')
    
    # Determine data root (parent of results directory)
    # Checkpoints are at: root/s0/best.ckpt
    # Data is at: root/../../data/... or absolute path
    # We'll use the parent of root's parent as data root
    data_root = os.path.abspath(os.path.join(root, '..', '..'))
    
    # Get test transform
    transform = get_test_transform()
    
    # Load ID dataset once (CIFAR-10)
    print(f'Loading ID dataset (CIFAR-10)...', file=sys.stderr)
    id_dataset = load_dataset('cifar10', data_root, transform)
    id_loader = DataLoader(
        id_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # CPU-safe
        pin_memory=False,
    )
    
    # Load OOD datasets
    ood_datasets = {}
    for ood_name in NEAR_OOD_DATASETS:
        print(f'Loading OOD dataset ({ood_name})...', file=sys.stderr)
        ood_datasets[ood_name] = load_dataset(ood_name, data_root, transform)
    
    # Store results per seed
    run_metrics = {}
    dataset_counts = {}
    
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        print(f'\nProcessing {seed_name}...', file=sys.stderr)
        
        # Load checkpoint
        checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}', file=sys.stderr)
            continue
        
        # Initialize model
        model = ResNet18_32x32(num_classes=10)
        
        # Load checkpoint (CPU-safe)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'net' in state_dict:
            state_dict = state_dict['net']
        model.load_state_dict(state_dict)
        model.eval()
        
        # Get ID scores
        print('  Computing ID scores...', file=sys.stderr)
        id_scores = []
        id_labels = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                labels = batch['label']
                logits = model(data)
                energy = compute_ebo_scores(logits, temperature)
                id_scores.append(energy.numpy())
                id_labels.append(labels.numpy())
        
        id_scores = np.concatenate(id_scores)
        id_labels = np.concatenate(id_labels)
        
        # Evaluate each OOD dataset
        seed_aurocs = {}
        for ood_name in NEAR_OOD_DATASETS:
            print(f'  Computing OOD scores ({ood_name})...', file=sys.stderr)
            ood_loader = DataLoader(
                ood_datasets[ood_name],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            
            ood_scores = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data']
                    logits = model(data)
                    energy = compute_ebo_scores(logits, temperature)
                    ood_scores.append(energy.numpy())
            
            ood_scores = np.concatenate(ood_scores)
            
            # Combine ID and OOD scores
            all_scores = np.concatenate([id_scores, ood_scores])
            all_labels = np.concatenate([
                id_labels,
                -np.ones(len(ood_scores), dtype=np.int64)
            ])
            
            # Compute AUROC
            auroc = compute_auroc(all_scores, all_labels)
            seed_aurocs[ood_name] = auroc * 100  # Convert to percentage
            
            # Store dataset counts
            if seed_name == seed_dirs[0]:  # Only count once
                dataset_counts[ood_name] = len(ood_scores)
        
        run_metrics[seed_name] = seed_aurocs
        print(f'  {seed_name} results: {seed_aurocs}', file=sys.stderr)
    
    # Compute dataset counts (from first seed)
    if not dataset_counts:
        dataset_counts = {ood: len(ood_datasets[ood]) for ood in NEAR_OOD_DATASETS}
    
    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-run dataset mean
    run_dataset_means = []
    for seed_name, seed_aurocs in run_metrics.items():
        dataset_mean = np.mean(list(seed_aurocs.values()))
        run_dataset_means.append(dataset_mean)
    
    # Then mean of runs
    actual = float(np.mean(run_dataset_means))
    
    # Build result dictionary
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    
    # Print the required JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')

if __name__ == '__main__':
    main()
