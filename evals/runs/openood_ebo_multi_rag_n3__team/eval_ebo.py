#!/usr/bin/env python3
"""Reproduce EBO Near-OOD AUROC for CIFAR-10 using OpenOOD checkpoints."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

# Direct imports from openood modules (no evaluators/postprocessors packages)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# Fix truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Constants - verified against OpenOOD source
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official benchmark image list paths (relative to data_root)
ID_IMGLIST = 'data/benchmark_imglist/cifar10/test.txt'
OOD_IMGLISTS = {
    'cifar100': 'data/benchmark_imglist/cifar10/nearood/cifar100.txt',
    'tin': 'data/benchmark_imglist/cifar10/nearood/tin.txt',
}

SEEDS = ['s0', 's1', 's2']

# ---------------------------------------------------------------------------
# Transform - verified against OpenOOD preprocessors/transform.py
# The official test pipeline for CIFAR-10 is:
#   Resize(32, interpolation=BILINEAR) -> CenterCrop(32) -> ToTensor() -> Normalize(mean, std)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the standard CIFAR-10 test transform used by OpenOOD."""
    return T.Compose([
        T.Resize(32, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score function (temperature=1)
# Verified against openood/postprocessors/ebo_postprocessor.py
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute EBO score: temperature * logsumexp(logits / temperature, dim=1)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation - verified against openood/evaluators/metrics.py
# Returns percentage (0-100)
# ---------------------------------------------------------------------------
def compute_auroc(conf_scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC in percentage points.
    
    Args:
        conf_scores: 1D array of confidence scores (higher = more ID-like)
        labels: 1D array where 1 = ID, 0 = OOD
    Returns:
        AUROC in percentage (0-100)
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, conf_scores)) * 100.0

# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_seed(net: torch.nn.Module, seed_dir: str, data_root: str,
                  batch_size: int, num_workers: int) -> dict:
    """Evaluate a single seed checkpoint on ID and OOD datasets.
    
    Returns:
        dict with keys 'cifar100' and 'tin' containing AUROC percentages.
    """
    device = next(net.parameters()).device
    transform = get_test_transform()

    # Load ID dataset (CIFAR-10 test)
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(data_root, ID_IMGLIST),
        data_dir=os.path.join(data_root, 'data', 'images_classic'),
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(
        id_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    # Collect ID scores
    net.eval()
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            images = batch['data'].to(device)
            logits = net(images)
            scores = ebo_score(logits).cpu().numpy()
            id_scores.append(scores)
    id_scores = np.concatenate(id_scores)

    results = {}
    for ood_name, imglist_rel in OOD_IMGLISTS.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_ood',
            imglist_pth=os.path.join(data_root, imglist_rel),
            data_dir=os.path.join(data_root, 'data', 'images_classic'),
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loader = DataLoader(
            ood_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )

        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                images = batch['data'].to(device)
                logits = net(images)
                scores = ebo_score(logits).cpu().numpy()
                ood_scores.append(scores)
        ood_scores = np.concatenate(ood_scores)

        # Build label array: ID=1, OOD=0
        labels = np.concatenate([
            np.ones(len(id_scores), dtype=np.int64),
            np.zeros(len(ood_scores), dtype=np.int64),
        ])
        confs = np.concatenate([id_scores, ood_scores])

        auroc = compute_auroc(confs, labels)
        results[ood_name] = auroc

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/s1/s2 subfolders')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    root = args.root
    # Determine data_root: typically the 'data' directory is at the same level as 'results'
    # Try common OpenOOD layout: root is results/cifar10_resnet18_32x32_base_e100_lr0.1_default
    # data is at the same level as results
    data_root = os.path.join(root, '..', '..', 'data')
    if not os.path.isdir(data_root):
        # Fallback: try relative to script
        data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    data_root = os.path.abspath(data_root)

    device = torch.device('cpu')

    # Verify checkpoint directories exist
    for seed in SEEDS:
        ckpt_path = os.path.join(root, seed, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}", file=sys.stderr)
            sys.exit(1)

    # Evaluate each seed
    run_metrics = {}
    dataset_counts = {}

    for seed in SEEDS:
        ckpt_path = os.path.join(root, seed, 'best.ckpt')
        net = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        seed_results = evaluate_seed(
            net, os.path.join(root, seed), data_root,
            args.batch_size, args.num_workers,
        )
        run_metrics[seed] = seed_results

        # Accumulate dataset counts (sample counts from OOD datasets)
        for dset_name in seed_results:
            if dset_name not in dataset_counts:
                # Count samples from the OOD dataset imglist
                imglist_path = os.path.join(data_root, OOD_IMGLISTS[dset_name])
                with open(imglist_path, 'r') as f:
                    n_samples = sum(1 for _ in f)
                dataset_counts[dset_name] = n_samples

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-run dataset mean
    run_dataset_means = []
    for seed in SEEDS:
        vals = list(run_metrics[seed].values())
        run_mean = float(np.mean(vals))
        run_dataset_means.append(run_mean)

    # Then mean of runs
    actual = float(np.mean(run_dataset_means))

    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(output)}')

if __name__ == '__main__':
    main()
