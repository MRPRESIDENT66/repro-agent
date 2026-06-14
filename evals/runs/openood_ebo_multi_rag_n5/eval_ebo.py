#!/usr/bin/env python3
"""CPU-safe EBO evaluation for OpenOOD ResNet18_32x32 on CIFAR-10.

Reproduces the official OpenOOD EBO evaluation for Near-OOD (CIFAR-100,
TinyImageNet) using the exact checkpoints, transforms, and dataset lists.
Prints a single JSON REPRO_RESULT line.
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 1. Model import (direct module, no evaluation_api)
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# 2. Dataset import (direct module, no evaluation_api)
# ---------------------------------------------------------------------------
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# 3. Constants (from OpenOOD transform.py and configs)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ImageNet normalization for TinyImageNet (same as OpenOOD uses)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# 4. Transform construction (exact OpenOOD test transform)
# ---------------------------------------------------------------------------
def build_test_transform(mean, std, pre_size=32, image_size=32):
    """Build the exact test transform from openood/preprocessors/transform.py
    and test_preprocessor.py: Convert('RGB') -> Resize(pre_size) ->
    CenterCrop(image_size) -> ToTensor() -> Normalize(mean, std)."""
    return tvs_trans.Compose([
        tvs_trans.Resize(pre_size, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(image_size),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])

# ---------------------------------------------------------------------------
# 5. EBO score function
# ---------------------------------------------------------------------------
def energy_score(logits, temperature=1.0):
    """Compute energy score: T * logsumexp(logits / T)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# 6. AUROC calculation (from scratch)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC given ID scores (higher = more ID-like) and OOD scores.
    Returns percentage (0-100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    
    # Sort by score descending (higher score = more ID-like)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    # True positive rate and false positive rate
    tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    
    # AUC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return auroc * 100.0  # convert to percentage

# ---------------------------------------------------------------------------
# 7. Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_run(net, id_loader, ood_loaders, device='cpu'):
    """Evaluate a single run (one seed checkpoint).
    
    Args:
        net: Loaded model in eval mode.
        id_loader: DataLoader for ID data (CIFAR-10 test).
        ood_loaders: dict mapping dataset name to DataLoader for OOD data.
        device: 'cpu' or 'cuda'.
    
    Returns:
        dict mapping OOD dataset name to AUROC percentage.
    """
    net.eval()
    
    # Collect ID energy scores (negative energy = higher = more ID-like)
    id_energies = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            logits = net(data)
            energies = energy_score(logits)
            id_energies.extend(-energies.cpu().numpy())  # negative: higher = more ID-like
    id_scores = np.array(id_energies)
    
    # Collect OOD energy scores for each dataset
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_energies = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                logits = net(data)
                energies = energy_score(logits)
                ood_energies.extend(-energies.cpu().numpy())
        ood_scores = np.array(ood_energies)
        
        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc
    
    return results

# ---------------------------------------------------------------------------
# 8. Dataset loading helpers
# ---------------------------------------------------------------------------
def load_imglist_dataset(imglist_path, data_dir, transform, name='dataset'):
    """Load an ImglistDataset with the given transform."""
    # We need a minimal config-like object for the preprocessor
    # Since we're not using TestStandardPreProcessor, we pass transform directly
    # But ImglistDataset expects preprocessor callable
    dataset = ImglistDataset(
        name=name,
        imglist_pth=imglist_path,
        data_dir=data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,  # same transform for aux
    )
    return dataset

def get_cifar10_test_loader(root_dir, batch_size=200, num_workers=0):
    """Load CIFAR-10 test set using OpenOOD's ImglistDataset."""
    imglist_path = os.path.join(root_dir, 'data', 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    data_dir = os.path.join(root_dir, 'data', 'images_classic')
    
    transform = build_test_transform(CIFAR10_MEAN, CIFAR10_STD)
    dataset = load_imglist_dataset(imglist_path, data_dir, transform, name='cifar10_test')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader

def get_ood_loader(dataset_name, root_dir, batch_size=200, num_workers=0):
    """Load an OOD dataset (cifar100 or tin) using OpenOOD's ImglistDataset."""
    if dataset_name == 'cifar100':
        imglist_path = os.path.join(root_dir, 'data', 'benchmark_imglist', 'cifar10', 'test_cifar100.txt')
        data_dir = os.path.join(root_dir, 'data', 'images_classic')
        mean, std = CIFAR10_MEAN, CIFAR10_STD  # CIFAR-100 uses same normalization as CIFAR-10
    elif dataset_name == 'tin':
        imglist_path = os.path.join(root_dir, 'data', 'benchmark_imglist', 'cifar10', 'test_tin.txt')
        data_dir = os.path.join(root_dir, 'data', 'images_classic')
        mean, std = IMAGENET_MEAN, IMAGENET_STD  # TinyImageNet uses ImageNet normalization
    else:
        raise ValueError(f"Unknown OOD dataset: {dataset_name}")
    
    transform = build_test_transform(mean, std)
    dataset = load_imglist_dataset(imglist_path, data_dir, transform, name=f'{dataset_name}_test')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader

# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='CPU-safe EBO evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/, s1/, s2/ subfolders')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    
    root = args.root
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Find checkpoint subfolders
    subfolders = sorted(glob(os.path.join(root, 's*')))
    if len(subfolders) == 0:
        print(f"Error: No subfolders found in {root}", file=sys.stderr)
        sys.exit(1)
    
    # Get OpenOOD root (parent of data/ directory)
    # Assume root is the results directory; data is at same level or specified
    # We'll try to find data relative to the script or root
    # For simplicity, assume data is at the same level as the results directory
    openood_root = os.path.dirname(os.path.abspath(__file__))
    # Data directory is at the same level as the script (repo root)
    data_root = os.path.join(openood_root, 'data')
    
    # Load ID data (CIFAR-10 test)
    print(f"Loading CIFAR-10 test set from {data_root}...", file=sys.stderr)
    id_loader = get_cifar10_test_loader(
        os.path.dirname(data_root),  # parent of data/
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Load OOD datasets
    ood_names = ['cifar100', 'tin']
    ood_loaders = {}
    for name in ood_names:
        print(f"Loading {name} OOD set...", file=sys.stderr)
        ood_loaders[name] = get_ood_loader(
            name,
            os.path.dirname(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    # Evaluate each seed
    run_metrics = {}
    dataset_counts = {}
    
    for subfolder in subfolders:
        seed_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...", file=sys.stderr)
            continue
        
        print(f"Evaluating {seed_name}...", file=sys.stderr)
        
        # Load model
        net = ResNet18_32x32(num_classes=NUM_CLASSES)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(state_dict)
        net.eval()
        
        # Evaluate
        results = evaluate_run(net, id_loader, ood_loaders, device='cpu')
        run_metrics[seed_name] = results
        
        # Track dataset counts (number of ID samples)
        if 'cifar100' not in dataset_counts:
            dataset_counts['cifar100'] = len(id_loader.dataset)
        if 'tin' not in dataset_counts:
            dataset_counts['tin'] = len(id_loader.dataset)
    
    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-dataset mean across runs
    dataset_means = {}
    for ood_name in ood_names:
        values = [run_metrics[seed][ood_name] for seed in run_metrics]
        dataset_means[ood_name] = np.mean(values)
    
    # Then mean of dataset means
    actual = np.mean(list(dataset_means.values()))
    
    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    
    # Print the required JSON line
    print(f"REPRO_RESULT {json.dumps(output)}")

if __name__ == '__main__':
    main()
