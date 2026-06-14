#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC evaluation for CIFAR-10.

Evaluates ResNet18_32x32 checkpoints from s0/s1/s2 on CIFAR-10 (ID) vs
CIFAR-100 and TinyImageNet (near-OOD). Uses EBO score (energy) and computes
AUROC locally. Prints a single JSON REPRO_RESULT line.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from sklearn import metrics
from torch.utils.data import DataLoader

# Direct imports from OpenOOD modules (no evaluators/postprocessors/evaluation_api)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
TEMPERATURE = 1.0

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ---------------------------------------------------------------------------
# Transform pipeline (exact reproduction of TestStandardPreProcessor)
# ---------------------------------------------------------------------------
def build_test_transform():
    """Build the test transform as in TestStandardPreProcessor."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Compute EBO (energy) score: temperature * logsumexp(logits / temperature)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (matching OpenOOD semantics)
# ---------------------------------------------------------------------------
def compute_auroc(energy_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC treating OOD as positive class.
    energy_scores: higher = more OOD-like.
    labels: -1 for OOD, otherwise ID class index.
    Returns AUROC as percentage (0-100).
    """
    ood_indicator = np.zeros_like(labels)
    ood_indicator[labels == -1] = 1
    # Negate energy because OpenOOD convention: higher conf = more ID-like
    fpr, tpr, _ = metrics.roc_curve(ood_indicator, -energy_scores)
    auroc = metrics.auc(fpr, tpr)
    return auroc * 100.0  # convert to percentage

# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model: torch.nn.Module, id_loader: DataLoader,
                        ood_loaders: dict) -> dict:
    """
    Evaluate a single checkpoint on ID and OOD datasets.
    Returns dict mapping dataset name -> AUROC (percentage).
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect ID energy scores
    id_energies = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            logits = model(data)
            energies = ebo_score(logits).cpu().numpy()
            id_energies.append(energies)
    id_energies = np.concatenate(id_energies)

    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_energies = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                logits = model(data)
                energies = ebo_score(logits).cpu().numpy()
                ood_energies.append(energies)
        ood_energies = np.concatenate(ood_energies)

        # Combine ID and OOD: ID labels = 0, OOD labels = -1
        all_energies = np.concatenate([id_energies, ood_energies])
        all_labels = np.concatenate([
            np.zeros(len(id_energies), dtype=int),
            -np.ones(len(ood_energies), dtype=int)
        ])

        auroc = compute_auroc(all_energies, all_labels)
        results[ood_name] = auroc

    return results

def main():
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC evaluation')
    parser.add_argument('--root', required=True,
                        help='Root directory containing s0/s1/s2 subfolders')
    args = parser.parse_args()

    root = args.root

    # Validate root structure
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        print(f'ERROR: No s* subfolders found in {root}', file=sys.stderr)
        sys.exit(1)

    # Build transform
    transform = build_test_transform()

    # Dataset paths (relative to root's parent? No, they are absolute paths
    # in the OpenOOD repo structure. We assume standard layout.)
    # We'll construct paths relative to a reasonable base.
    # The benchmark image lists are typically at:
    # ./data/benchmark_imglist/cifar10/
    # We'll assume they exist relative to CWD or we can accept a --data-root.
    # For simplicity, use standard OpenOOD paths.
    data_root = './data/images_classic'
    imglist_root = './data/benchmark_imglist/cifar10'

    # ID dataset: CIFAR-10 test
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(imglist_root, 'test_cifar10.txt'),
        data_dir=data_root,
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform
    )
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=False)

    # OOD datasets
    ood_configs = {
        'cifar100': 'test_cifar100.txt',
        'tin': 'test_tin.txt',
    }
    ood_loaders = {}
    for name, list_file in ood_configs.items():
        dataset = ImglistDataset(
            name=name,
            imglist_pth=os.path.join(imglist_root, list_file),
            data_dir=data_root,
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform
        )
        ood_loaders[name] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False
        )

    # Store per-seed, per-dataset AUROCs
    run_metrics = {}
    dataset_aurocs = {'cifar100': [], 'tin': []}

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        checkpoint_path = os.path.join(seed_dir, 'best.ckpt')

        if not os.path.isfile(checkpoint_path):
            print(f'WARNING: Checkpoint not found: {checkpoint_path}', file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()  # Keep on CPU

        # Evaluate
        results = evaluate_checkpoint(model, id_loader, ood_loaders)

        run_metrics[seed_name] = {}
        for ds_name, auroc in results.items():
            run_metrics[seed_name][ds_name] = round(auroc, 2)
            dataset_aurocs[ds_name].append(auroc)

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-run dataset mean
    per_run_dataset_means = []
    for seed_name in sorted(run_metrics.keys()):
        ds_values = [run_metrics[seed_name][ds] for ds in ['cifar100', 'tin']]
        per_run_dataset_means.append(np.mean(ds_values))

    # Then mean of runs
    actual = float(np.mean(per_run_dataset_means))

    # Build output
    # datasets: evaluated sample counts (we don't have them directly, but we
    # can compute from dataset lengths)
    dataset_counts = {
        'cifar100': len(ood_loaders['cifar100'].dataset),
        'tin': len(ood_loaders['tin'].dataset),
    }

    output = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 2),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print(f'REPRO_RESULT {json.dumps(output)}')

if __name__ == '__main__':
    main()
