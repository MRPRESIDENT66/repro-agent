#!/usr/bin/env python3
"""eval_ebo.py - CPU-safe EBO evaluation for CIFAR-10 near-OOD AUROC.

Implements the exact OpenOOD evaluation protocol using:
- ResNet18_32x32 from openood.networks.resnet18_32x32
- ImglistDataset from openood.datasets.imglist_dataset
- EBO postprocessor logic from openood/postprocessors/ebo_postprocessor.py
- Official checkpoints at results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s{0,1,2}/best.ckpt
- Official benchmark image lists for CIFAR-100 and TinyImageNet near-OOD

Prints a single JSON line: REPRO_RESULT {...}
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, or postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC using official OpenOOD convention.

    The EBO postprocessor returns energy scores where higher energy = more OOD-like.
    The official OpenOOD metrics.py negates confidence before calling
    sklearn.metrics.roc_curve, because roc_curve expects higher scores = more ID-like.
    We replicate that logic here.
    """
    from sklearn import metrics
    # OOD indicator: 1 for OOD, 0 for ID
    ood_indicator = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # Concatenate scores (higher = more OOD-like)
    conf = np.concatenate([id_scores, ood_scores])
    # Negate conf as in official code: roc_curve(ood_indicator, -conf)
    fpr_list, tpr_list, _ = metrics.roc_curve(ood_indicator, -conf)
    auroc = metrics.auc(fpr_list, tpr_list)
    return auroc


def get_test_transform():
    """CIFAR-10 test transform from openood/preprocessors/transform.py.
    
    Uses the exact normalization parameters from the repository:
    mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]
    """
    return transforms.Compose([
        transforms.Resize(32, interpolation=Image.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.2435, 0.2616]),
    ])


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load ResNet18_32x32 from checkpoint."""
    net = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def get_dataset(imglist_pth, data_dir, transform):
    """Create ImglistDataset with given transform."""
    return ImglistDataset(
        name='test',
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )


def evaluate_run(net, id_loader, ood_loaders, device='cpu', temperature=1.0):
    """Evaluate a single run (one seed) and return per-dataset AUROC."""
    net = net.to(device)
    
    # Collect ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            output = net(data)
            # EBO score: temperature * logsumexp(logits / temperature)
            energy = temperature * torch.logsumexp(output / temperature, dim=1)
            id_scores.append(energy.cpu().numpy())
    id_scores = np.concatenate(id_scores)
    
    # Collect OOD scores for each dataset
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                output = net(data)
                energy = temperature * torch.logsumexp(output / temperature, dim=1)
                ood_scores.append(energy.cpu().numpy())
        ood_scores = np.concatenate(ood_scores)
        
        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc * 100  # Convert to percentage
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    args = parser.parse_args()
    
    root = args.root
    device = 'cpu'
    temperature = 1.0  # Default EBO temperature
    
    # Checkpoint paths
    checkpoint_paths = {
        's0': os.path.join(root, 's0', 'best.ckpt'),
        's1': os.path.join(root, 's1', 'best.ckpt'),
        's2': os.path.join(root, 's2', 'best.ckpt'),
    }
    
    # Verify checkpoints exist
    for seed, path in checkpoint_paths.items():
        if not os.path.isfile(path):
            print(f"Error: Checkpoint not found at {path}", file=sys.stderr)
            sys.exit(1)
    
    # Data paths (from configs/datasets/cifar10/cifar10_ood.yml)
    data_dir = './data/images_classic/'
    id_imglist = './data/benchmark_imglist/cifar10/test_cifar10.txt'
    ood_imglists = {
        'cifar100': './data/benchmark_imglist/cifar10/test_cifar100.txt',
        'tin': './data/benchmark_imglist/cifar10/test_tin.txt',
    }
    
    # Verify data files exist
    for path in [id_imglist] + list(ood_imglists.values()):
        if not os.path.isfile(path):
            print(f"Error: Image list not found at {path}", file=sys.stderr)
            sys.exit(1)
    
    # Create transform
    transform = get_test_transform()
    
    # Create datasets
    id_dataset = get_dataset(id_imglist, data_dir, transform)
    ood_datasets = {}
    for name, imglist in ood_imglists.items():
        ood_datasets[name] = get_dataset(imglist, data_dir, transform)
    
    # Create data loaders
    batch_size = 200
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_loaders = {}
    for name, dataset in ood_datasets.items():
        ood_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate each run
    run_metrics = {}
    dataset_counts = {}
    
    for seed in ['s0', 's1', 's2']:
        net = load_checkpoint(checkpoint_paths[seed], device=device)
        results = evaluate_run(net, id_loader, ood_loaders, device=device, temperature=temperature)
        run_metrics[seed] = results
        
        # Record dataset counts (from first run, they're the same for all runs)
        if not dataset_counts:
            for name in ood_datasets:
                dataset_counts[name] = len(ood_datasets[name])
    
    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-dataset mean across runs
    dataset_means = {}
    for dataset_name in ood_datasets:
        values = [run_metrics[seed][dataset_name] for seed in ['s0', 's1', 's2']]
        dataset_means[dataset_name] = np.mean(values)
    
    # Then compute mean of dataset means (dataset_mean_then_run_mean)
    actual = np.mean(list(dataset_means.values()))
    
    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    
    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
