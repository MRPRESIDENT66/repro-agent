#!/usr/bin/env python3
"""
eval_ebo.py - Reproduce EBO Near-OOD AUROC for CIFAR-10 using OpenOOD checkpoints.

Usage:
    python eval_ebo.py --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default

Prints a single JSON line: REPRO_RESULT {...}
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn import metrics
from torch.utils.data import DataLoader

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, or postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 0  # CPU-safe, avoid shared memory issues

# CIFAR-10 normalization from openood/preprocessors/transform.py
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ---------------------------------------------------------------------------
# Transform: directly from openood/preprocessors/transform.py
# TestStandardPreProcessor uses: ToTensor -> Normalize
# ---------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Dataset paths (relative to data root)
# ---------------------------------------------------------------------------
# ID: CIFAR-10 test set
ID_IMGLIST = './data/benchmark_imglist/cifar10/test_cifar10.txt'
ID_DATA_DIR = './data/images_classic'

# Near-OOD: CIFAR-100
OOD_CIFAR100_IMGLIST = './data/benchmark_imglist/cifar100/test_cifar100.txt'
OOD_CIFAR100_DATA_DIR = './data/images_classic'

# Near-OOD: TinyImageNet (resized to 32x32)
OOD_TIN_IMGLIST = './data/benchmark_imglist/cifar100/test_tin.txt'
OOD_TIN_DATA_DIR = './data/images_classic'


def create_dataloader(imglist_pth, data_dir, num_classes=NUM_CLASSES):
    """Create a DataLoader for a given image list."""
    dataset = ImglistDataset(
        name='test',
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=num_classes,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return loader


def compute_ebo_scores(logits, temperature=1.0):
    """Compute EBO (negative energy) scores.
    
    EBO score = -E(x) = -T * log(Σ exp(f_i(x)/T))
    Higher score = more OOD-like.
    """
    # logits shape: (N, num_classes)
    # Use logsumexp for numerical stability
    energy = temperature * torch.logsumexp(logits / temperature, dim=1)
    # Negative energy: higher = more OOD
    return -energy


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC treating OOD as positive class.
    
    Following OpenOOD convention: OOD indicator = 1 for OOD, 0 for ID.
    The scores are such that higher = more OOD-like.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)
    return auroc * 100.0  # Convert to percentage


def evaluate_run(checkpoint_path, id_loader, ood_loaders):
    """Evaluate a single run (s0, s1, or s2)."""
    # Load model
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Handle potential 'net.' prefix in state dict keys
    if any(k.startswith('net.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[4:] if k.startswith('net.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.eval()

    # Inference function
    def get_scores(loader):
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data']
                logits = model(data)
                scores = compute_ebo_scores(logits)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)

    # ID scores
    id_scores = get_scores(id_loader)

    # OOD scores per dataset
    results = {}
    for name, loader in ood_loaders.items():
        ood_scores = get_scores(loader)
        auroc = compute_auroc(id_scores, ood_scores)
        results[name] = auroc

    return results, len(id_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Path to checkpoint root directory (contains s0, s1, s2)')
    args = parser.parse_args()

    root = args.root

    # Verify checkpoint structure
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if len(subfolders) == 0:
        print(f'Error: No subfolders found in {root}', file=sys.stderr)
        sys.exit(1)

    # Create datasets once (they are the same for all runs)
    print('Loading datasets...', file=sys.stderr)
    id_loader = create_dataloader(ID_IMGLIST, ID_DATA_DIR, num_classes=10)
    ood_loaders = {
        'cifar100': create_dataloader(OOD_CIFAR100_IMGLIST, OOD_CIFAR100_DATA_DIR, num_classes=100),
        'tin': create_dataloader(OOD_TIN_IMGLIST, OOD_TIN_DATA_DIR, num_classes=200),
    }

    # Evaluate each run
    run_metrics = {}
    dataset_counts = {}
    first_run = True

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        
        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}', file=sys.stderr)
            continue

        print(f'Evaluating {run_name}...', file=sys.stderr)
        results, id_count = evaluate_run(checkpoint_path, id_loader, ood_loaders)
        
        run_metrics[run_name] = results
        
        if first_run:
            # Get dataset counts (same for all runs)
            for name, loader in ood_loaders.items():
                dataset_counts[name] = len(loader.dataset)
            dataset_counts['cifar100'] = len(ood_loaders['cifar100'].dataset)
            dataset_counts['tin'] = len(ood_loaders['tin'].dataset)
            first_run = False

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-dataset mean across runs
    dataset_means = {}
    for dataset_name in ['cifar100', 'tin']:
        values = [run_metrics[run][dataset_name] for run in run_metrics]
        dataset_means[dataset_name] = np.mean(values)

    # Then, mean of dataset means = mean across all dataset values
    all_values = [v for v in dataset_means.values()]
    actual = np.mean(all_values)

    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print the required JSON line
    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
