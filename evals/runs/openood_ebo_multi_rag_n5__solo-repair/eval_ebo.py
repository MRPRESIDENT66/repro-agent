#!/usr/bin/env python3
"""
Reproduce official OpenOOD EBO Near-OOD AUROC for CIFAR-10.
CPU-only, offline, using official checkpoints and datasets.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

# ---------------------------------------------------------------------------
# Import only the required modules from openood – no evaluators/postprocessors
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_WORKERS = 4

# CIFAR-10 test transform (from openood/preprocessors/transform.py + test_preprocessor.py)
# Resize(32) -> CenterCrop(32) -> ToTensor -> Normalize(cifar10 mean, std)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Helper: build dataset from image list
# ---------------------------------------------------------------------------
def build_dataset(imglist_rel_path, data_dir):
    """Return an ImglistDataset with the CIFAR-10 test transform."""
    # The preprocessor argument is the transform callable.
    # data_aux_preprocessor is unused for test, pass same transform.
    return ImglistDataset(
        name='eval',
        imglist_pth=imglist_rel_path,
        data_dir=data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )

# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score: -logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Return AUROC in percentage points."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending (higher energy = more OOD-like)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0
    # TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing checkpoints and data')
    args = parser.parse_args()
    root = args.root

    # -----------------------------------------------------------------------
    # Paths (assume standard OpenOOD layout under root)
    # -----------------------------------------------------------------------
    # Checkpoints: s0/best.ckpt, s1/best.ckpt, s2/best.ckpt
    ckpt_dir = root
    # Image lists: we assume they are under root/data/...
    # Typical OpenOOD locations:
    #   ID:  ./data/benchmark_imglist/cifar10/test_cifar10.txt
    #   OOD: ./data/benchmark_imglist/cifar100/test_cifar100.txt, ./data/benchmark_imglist/tinyimagenet/test_tinyimagenet.txt
    data_dir = os.path.join(root, '..', '..', 'data', 'images_classic')
    id_list = os.path.join(data_dir, 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    ood_lists = {
        'cifar100': os.path.join(data_dir, 'benchmark_imglist', 'cifar100', 'test_cifar100.txt'),
        'tin': os.path.join(data_dir, 'benchmark_imglist', 'tinyimagenet', 'test_tinyimagenet.txt'),
    }

    # Verify existence
    for p in [ckpt_dir, id_list] + list(ood_lists.values()):
        if not os.path.exists(p):
            print(f"ERROR: required path not found: {p}", file=sys.stderr)
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Build ID dataset (CIFAR-10 test)
    # -----------------------------------------------------------------------
    id_dataset = build_dataset(id_list, data_dir)
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS)

    # -----------------------------------------------------------------------
    # Build OOD datasets
    # -----------------------------------------------------------------------
    ood_datasets = {}
    ood_loaders = {}
    for name, lst in ood_lists.items():
        ds = build_dataset(lst, data_dir)
        ood_datasets[name] = ds
        ood_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=NUM_WORKERS)

    # -----------------------------------------------------------------------
    # Run evaluation for each checkpoint
    # -----------------------------------------------------------------------
    run_keys = ['s0', 's1', 's2']
    # Store per-run, per-dataset AUROC
    run_metrics = {rk: {} for rk in run_keys}
    # Store sample counts per dataset (should be same across runs)
    dataset_counts = {}

    device = torch.device('cpu')

    for run_key in run_keys:
        ckpt_path = os.path.join(ckpt_dir, run_key, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location=device)
        # state may be a dict with 'state_dict' or directly the state_dict
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        model.eval()

        # ID scores
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = ebo_score(logits).cpu().numpy()
                id_scores_list.append(scores)
        id_scores = np.concatenate(id_scores_list)
        # Record count (first run only)
        if run_key == 's0':
            dataset_counts['cifar10'] = len(id_scores)

        # OOD scores per dataset
        for ood_name, loader in ood_loaders.items():
            ood_scores_list = []
            with torch.no_grad():
                for batch in loader:
                    images = batch['data'].to(device)
                    logits = model(images)
                    scores = ebo_score(logits).cpu().numpy()
                    ood_scores_list.append(scores)
            ood_scores = np.concatenate(ood_scores_list)
            if run_key == 's0':
                dataset_counts[ood_name] = len(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_key][ood_name] = auroc

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # For each run, compute mean over datasets (cifar100, tin)
    run_means = []
    for rk in run_keys:
        vals = list(run_metrics[rk].values())
        run_mean = float(np.mean(vals))
        run_means.append(run_mean)
    actual = float(np.mean(run_means))

    # Build output
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': {
            'cifar100': dataset_counts.get('cifar100', 0),
            'tin': dataset_counts.get('tin', 0),
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
