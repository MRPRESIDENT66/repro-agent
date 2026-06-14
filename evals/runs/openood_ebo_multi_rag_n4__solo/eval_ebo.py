#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD checkpoints."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Import only the required modules from openood
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official checkpoint paths (relative to --root)
CHECKPOINT_PATHS = {
    's0': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    's1': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt',
    's2': 'cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt',
}

# Benchmark image list paths (relative to --root)
BENCHMARK_DIR = 'data/benchmark_imglist'
ID_LIST = os.path.join(BENCHMARK_DIR, 'cifar10/test_cifar10.txt')
OOD_LISTS = {
    'cifar100': os.path.join(BENCHMARK_DIR, 'cifar100/test_cifar100.txt'),
    'tin': os.path.join(BENCHMARK_DIR, 'tin/test_tin.txt'),
}

# ---------------------------------------------------------------------------
# Transform (exact copy of TestStandardPreProcessor pipeline)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the exact test transform from openood/preprocessors/transform.py."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Energy-Based OOD score: -logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points (higher = better OOD detection)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    # Sort by score descending (higher energy = more OOD-like)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 50.0  # random performance

    # Compute TPR and FPR
    tpr = np.cumsum(sorted_labels == 1) / pos_count
    fpr = np.cumsum(sorted_labels == 0) / neg_count

    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)  # convert to percentage


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def evaluate_checkpoint(root: str, checkpoint_path: str, device: torch.device) -> dict:
    """Evaluate a single checkpoint on ID (CIFAR-10 test) and OOD datasets."""
    # Load model
    model = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(os.path.join(root, checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_test_transform()

    # ID dataset (CIFAR-10 test)
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(root, ID_LIST),
        data_dir=os.path.join(root, 'data/images_classic'),
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=None,
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)

    # Collect ID scores
    id_scores_list = []
    with torch.no_grad():
        for batch in id_loader:
            images = batch['data'].to(device)
            logits = model(images)
            scores = compute_ebo_scores(logits)
            id_scores_list.append(scores.cpu().numpy())
    id_scores = np.concatenate(id_scores_list)

    results = {}
    for ood_name, ood_list_path in OOD_LISTS.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=os.path.join(root, ood_list_path),
            data_dir=os.path.join(root, 'data/images_classic'),
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=None,
        )
        ood_loader = DataLoader(ood_dataset, batch_size=200, shuffle=False, num_workers=0)

        ood_scores_list = []
        with torch.no_grad():
            for batch in ood_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = compute_ebo_scores(logits)
                ood_scores_list.append(scores.cpu().numpy())
        ood_scores = np.concatenate(ood_scores_list)

        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.',
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Evaluate all three checkpoints
    run_metrics = {}
    for run_name, ckpt_rel_path in CHECKPOINT_PATHS.items():
        print(f'Evaluating {run_name}...', file=sys.stderr)
        results = evaluate_checkpoint(args.root, ckpt_rel_path, device)
        run_metrics[run_name] = results

    # Compute dataset means per run, then mean of runs
    dataset_names = list(OOD_LISTS.keys())
    dataset_means = {d: [] for d in dataset_names}
    for run_name, metrics in run_metrics.items():
        for d in dataset_names:
            dataset_means[d].append(metrics[d])

    # Dataset mean across runs
    dataset_avg = {d: float(np.mean(dataset_means[d])) for d in dataset_names}

    # Final actual = mean of dataset averages
    actual = float(np.mean(list(dataset_avg.values())))

    # Count evaluated samples (ID + OOD per run, but we report dataset counts)
    # Each dataset is evaluated once per run, but we report the number of datasets
    dataset_counts = {d: 1 for d in dataset_names}  # each dataset appears once

    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
