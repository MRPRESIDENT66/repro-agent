#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC evaluation for CIFAR-10.

Reproduces the exact OpenOOD evaluation using ResNet18_32x32 checkpoints
(s0/s1/s2) and official CIFAR-10 preprocessing. Prints the required
REPRO_RESULT JSON line.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Only import what we need from openood, avoiding the faiss dependency chain
from openood.networks import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor


# ---------------------------------------------------------------------------
# Custom preprocessor matching OpenOOD's base_preprocessor for CIFAR-10
# ---------------------------------------------------------------------------
class CIFAR10Preprocessor(BasePreprocessor):
    """Standard CIFAR-10 preprocessing (normalization only)."""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
    
    def setup(self, config=None):
        pass
    
    def preprocess(self, image):
        return self.transform(image)


# ---------------------------------------------------------------------------
# Minimal EBO postprocessor (no OpenOOD package dependency)
# ---------------------------------------------------------------------------
class EBOPostprocessor:
    """Energy-based OOD detection postprocessor (temperature=1)."""
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    @torch.no_grad()
    def postprocess(self, net, data):
        logits = net(data)
        # Energy score = temperature * logsumexp(logits / temperature)
        energy = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        # Higher energy -> more OOD-like
        return energy.cpu().numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_dataloaders(data_root='./data', batch_size=200):
    """Get ID and OOD dataloaders."""
    preprocessor = CIFAR10Preprocessor()
    
    # ID test loader (CIFAR-10) - load from pre-existing offline path
    # The CIFAR-10 test set is stored as individual PNG files in images_classic/
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(data_root, 
            'benchmark_imglist/cifar10/test_cifar10.txt'),
        data_dir=os.path.join(data_root, 'images_classic'),
        num_classes=10,
        preprocessor=preprocessor
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    
    # OOD loaders using OpenOOD's ImglistDataset
    ood_loaders = {}
    
    # CIFAR-100
    cifar100_dataset = ImglistDataset(
        name='cifar10_ood_cifar100',
        imglist_pth=os.path.join(data_root, 
            'benchmark_imglist/cifar10/test_cifar100.txt'),
        data_dir=os.path.join(data_root, 'images_classic'),
        num_classes=10,
        preprocessor=preprocessor
    )
    ood_loaders['cifar100'] = DataLoader(cifar100_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    
    # TinyImageNet
    tin_dataset = ImglistDataset(
        name='cifar10_ood_tin',
        imglist_pth=os.path.join(data_root,
            'benchmark_imglist/cifar10/test_tin.txt'),
        data_dir=os.path.join(data_root, 'images_classic'),
        num_classes=10,
        preprocessor=preprocessor
    )
    ood_loaders['tin'] = DataLoader(tin_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0)
    
    return id_loader, ood_loaders


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC (percentage) from ID and OOD energy scores.
    Higher energy -> more OOD-like.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    # Sort by score descending (higher score = more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    # True positive rate and false positive rate
    tpr = np.cumsum(labels_sorted) / np.sum(labels_sorted)
    fpr = np.cumsum(1 - labels_sorted) / np.sum(1 - labels_sorted)
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr) * 100.0
    return auroc


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_run(net, id_loader, ood_loaders, device='cpu'):
    """Evaluate a single run (one checkpoint). Returns dict of AUROC per OOD dataset."""
    net.eval()
    postprocessor = EBOPostprocessor(temperature=1.0)

    # ID scores
    id_scores = []
    for batch, _ in id_loader:
        batch = batch.to(device)
        scores = postprocessor.postprocess(net, batch)
        id_scores.append(scores)
    id_scores = np.concatenate(id_scores)

    # OOD scores per dataset
    results = {}
    for name, loader in ood_loaders.items():
        ood_scores = []
        for batch, _ in loader:
            batch = batch.to(device)
            scores = postprocessor.postprocess(net, batch)
            ood_scores.append(scores)
        ood_scores = np.concatenate(ood_scores)
        results[name] = compute_auroc(id_scores, ood_scores)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root (contains s0/, s1/, s2/)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    device = 'cpu'

    # Load data
    print("Loading datasets...")
    id_loader, ood_loaders = get_dataloaders(args.data_root, args.batch_size)

    # Get dataset sizes
    dataset_sizes = {
        'cifar100': len(ood_loaders['cifar100'].dataset),
        'tin': len(ood_loaders['tin'].dataset),
    }

    # Iterate over runs (s0, s1, s2)
    run_dirs = ['s0', 's1', 's2']
    run_metrics = {}

    for run_name in run_dirs:
        ckpt_path = os.path.join(args.root, run_name, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        print(f"Loading checkpoint {ckpt_path}...")
        net = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)

        print(f"Evaluating run {run_name}...")
        results = evaluate_run(net, id_loader, ood_loaders, device)
        run_metrics[run_name] = results
        print(f"  {run_name}: cifar100={results['cifar100']:.2f}, tin={results['tin']:.2f}")

    # Compute dataset mean within each run, then mean of runs
    dataset_means = {}
    for run_name in run_dirs:
        dataset_means[run_name] = np.mean(list(run_metrics[run_name].values()))

    actual = np.mean(list(dataset_means.values()))

    # Build output
    output = {
        "metric": "near_ood_auroc",
        "actual": float(f"{actual:.2f}"),
        "datasets": dataset_sizes,
        "run_metrics": {
            run_name: {
                "cifar100": float(f"{run_metrics[run_name]['cifar100']:.2f}"),
                "tin": float(f"{run_metrics[run_name]['tin']:.2f}"),
            }
            for run_name in run_dirs
        },
        "aggregation": "dataset_mean_then_run_mean"
    }

    print(f"REPRO_RESULT {json.dumps(output)}")


if __name__ == '__main__':
    main()
