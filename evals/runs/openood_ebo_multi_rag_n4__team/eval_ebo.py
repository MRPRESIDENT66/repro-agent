#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe reproduction of OpenOOD EBO evaluation for ResNet18_32x32
on CIFAR-10 ID / CIFAR-100 & TinyImageNet near-OOD.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default

Prints exactly one JSON line: REPRO_RESULT {...}
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
from PIL import Image
from torch.utils.data import DataLoader

# Direct imports from openood (no evaluation_api, evaluators, or postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants (from OpenOOD configs for CIFAR-10 base preprocessor)
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
PRE_SIZE = 32
IMAGE_SIZE = 32
# Exact CIFAR-10 normalization from openood/preprocessors/transform.py
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]
BATCH_SIZE = 200
TEMPERATURE = 1.0

# ---------------------------------------------------------------------------
# Transform pipeline (exact copy of TestStandardPreProcessor for CIFAR-10)
# ---------------------------------------------------------------------------
test_transform = tvs_trans.Compose([
    tvs_trans.Resize(PRE_SIZE, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(IMAGE_SIZE),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=MEAN, std=STD),
])


# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Negative free energy: -E(x) = logsumexp(f(x)/T) * T"""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    # Sort by score descending (higher score -> more ID-like)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)

    if pos_count == 0 or neg_count == 0:
        return 50.0

    # TPR and FPR
    tpr = np.cumsum(sorted_labels == 1) / pos_count
    fpr = np.cumsum(sorted_labels == 0) / neg_count

    # AUC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_seed(seed_dir: str, data_root: str) -> dict:
    """Evaluate a single seed checkpoint and return per-dataset AUROC."""
    # Load model
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    ckpt_path = os.path.join(seed_dir, 'best.ckpt')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------------------------------------------------
    # ID dataset: CIFAR-10 test
    # -----------------------------------------------------------------------
    id_imglist = os.path.join(data_root, 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    id_data_dir = os.path.join(data_root, 'images_classic')
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Collect ID energy scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            images = batch['data']
            logits = model(images)
            scores = energy_score(logits)
            id_scores.append(scores.cpu().numpy())
    id_scores = np.concatenate(id_scores)

    # -----------------------------------------------------------------------
    # OOD datasets
    # -----------------------------------------------------------------------
    ood_configs = [
        ('cifar100', 'test_cifar100.txt'),
        ('tin', 'test_tin.txt'),
    ]

    results = {}
    for ood_name, imglist_file in ood_configs:
        ood_imglist = os.path.join(data_root, 'benchmark_imglist', 'cifar10', imglist_file)
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=ood_imglist,
            data_dir=id_data_dir,  # same image root for CIFAR-100; TinyImageNet uses same dir
            num_classes=NUM_CLASSES,
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform,
        )
        ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                images = batch['data']
                logits = model(images)
                scores = energy_score(logits)
                ood_scores.append(scores.cpu().numpy())
        ood_scores = np.concatenate(ood_scores)

        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/, s1/, s2/ subfolders')
    args = parser.parse_args()

    root = args.root
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    data_root = os.path.abspath(data_root)

    # Discover seed directories
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        print(f'ERROR: No seed directories found in {root}', file=sys.stderr)
        sys.exit(1)

    # Evaluate each seed
    all_run_metrics = {}
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        metrics = evaluate_seed(seed_dir, data_root)
        all_run_metrics[seed_name] = metrics

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    datasets = ['cifar100', 'tin']
    dataset_counts = {
        'cifar100': len(ImglistDataset(
            name='cifar100_test',
            imglist_pth=os.path.join(data_root, 'benchmark_imglist', 'cifar10', 'test_cifar100.txt'),
            data_dir=os.path.join(data_root, 'images_classic'),
            num_classes=NUM_CLASSES,
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform,
        )),
        'tin': len(ImglistDataset(
            name='tin_test',
            imglist_pth=os.path.join(data_root, 'benchmark_imglist', 'cifar10', 'test_tin.txt'),
            data_dir=os.path.join(data_root, 'images_classic'),
            num_classes=NUM_CLASSES,
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform,
        )),
    }

    # Per-run dataset mean
    run_means = []
    for seed_name, metrics in all_run_metrics.items():
        ds_values = [metrics[ds] for ds in datasets]
        run_mean = float(np.mean(ds_values))
        run_means.append(run_mean)

    # Final actual = mean of run means
    actual = float(np.mean(run_means))

    # Build output
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': all_run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
