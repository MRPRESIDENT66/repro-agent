#!/usr/bin/env python3
"""
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32
checkpoints s0, s1, s2 and Near-OOD datasets CIFAR-100 and TinyImageNet.
CPU-only, offline. Prints single JSON REPRO_RESULT line.
"""

import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

# Direct imports from openood modules (no evaluation_api, evaluators, postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 4

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Checkpoint paths relative to root
CHECKPOINT_REL = {
    's0': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    's1': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt',
    's2': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt',
}

# Near-OOD datasets
NEAR_OOD_DATASETS = ['cifar100', 'tin']

# ---------------------------------------------------------------------------
# Transform pipeline (exactly as in TestStandardPreProcessor)
# ---------------------------------------------------------------------------
def build_test_transform():
    """Build the test transform: Convert('RGB'), Resize(32), CenterCrop(32),
    ToTensor, Normalize(cifar10)."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(root: str, dataset_name: str, split: str, transform):
    """Load an ImglistDataset for the given dataset and split."""
    if dataset_name == 'cifar10':
        imglist_pth = os.path.join(root, 'data/benchmark_imglist/cifar10', f'{split}_cifar10.txt')
        data_dir = os.path.join(root, 'data/images_classic/')
    elif dataset_name == 'cifar100':
        imglist_pth = os.path.join(root, 'data/benchmark_imglist/cifar100', f'{split}_cifar100.txt')
        data_dir = os.path.join(root, 'data/images_classic/')
    elif dataset_name == 'tin':
        imglist_pth = os.path.join(root, 'data/benchmark_imglist/tin', f'{split}_tin.txt')
        data_dir = os.path.join(root, 'data/images_classic/')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # We need a preprocessor object that is callable. Use a simple wrapper.
    class PreprocessorWrapper:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, img):
            return self.transform(img)

    preprocessor = PreprocessorWrapper(transform)

    dataset = ImglistDataset(
        name=dataset_name + '_' + split,
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=preprocessor,
        data_aux_preprocessor=preprocessor,  # dummy, not used for test
    )
    return dataset

# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Energy-Based OOD score: -T * logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC (Area Under the Receiver Operating Characteristic curve).
    Returns percentage (0-100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])

    # Sort by score descending (higher score = more OOD-like for energy)
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    if pos == 0 or neg == 0:
        return 50.0  # random

    # Compute TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg

    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return auroc * 100.0  # percentage

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    root = args.root

    # Build transform
    transform = build_test_transform()

    # Load ID dataset (CIFAR-10 test)
    print('Loading CIFAR-10 test set...', file=sys.stderr)
    id_dataset = load_dataset(root, 'cifar10', 'test', transform)
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS)

    # Load OOD datasets
    ood_loaders = {}
    for ood_name in NEAR_OOD_DATASETS:
        print(f'Loading {ood_name} test set...', file=sys.stderr)
        ood_dataset = load_dataset(root, ood_name, 'test', transform)
        ood_loaders[ood_name] = DataLoader(ood_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=NUM_WORKERS)

    # Results storage
    run_metrics = {}
    dataset_counts = {}

    for run_name in ['s0', 's1', 's2']:
        print(f'\nEvaluating run {run_name}...', file=sys.stderr)

        # Load checkpoint
        ckpt_path = os.path.join(root, 'results', CHECKPOINT_REL[run_name])
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)

        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # Handle possible 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data']
                logits = model(images)
                scores = compute_ebo_scores(logits)
                id_scores.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores)
        dataset_counts['cifar100'] = len(id_scores)  # placeholder, will be updated

        # Compute OOD scores for each dataset
        ood_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch['data']
                    logits = model(images)
                    scores = compute_ebo_scores(logits)
                    ood_scores.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores)

            # Compute AUROC
            auroc = compute_auroc(id_scores, ood_scores)
            ood_results[ood_name] = auroc
            dataset_counts[ood_name] = len(ood_scores)

        run_metrics[run_name] = ood_results
        print(f'  {run_name}: {ood_results}', file=sys.stderr)

    # Update dataset_counts with ID count
    dataset_counts['cifar100'] = len(id_scores)  # ID count for cifar100 (same for all runs)

    # Compute aggregation: dataset mean within each run, then mean of runs
    # For each run, compute mean AUROC across datasets
    run_means = []
    for run_name in ['s0', 's1', 's2']:
        metrics = run_metrics[run_name]
        mean_auroc = np.mean([metrics[ds] for ds in NEAR_OOD_DATASETS])
        run_means.append(mean_auroc)

    actual = np.mean(run_means)

    # Build result dict
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': dataset_counts.get('cifar100', 0),
            'tin': dataset_counts.get('tin', 0),
        },
        'run_metrics': {
            run: {
                ds: float(run_metrics[run][ds])
                for ds in NEAR_OOD_DATASETS
            }
            for run in ['s0', 's1', 's2']
        },
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
