#!/usr/bin/env python3
"""CPU-safe EBO evaluation for OpenOOD ResNet18_32x32 on CIFAR-10.

Implements the exact evaluation protocol from OpenOOD's eval_ood.py but
without importing evaluation_api, evaluators, or postprocessors.
Computes EBO scores and AUROC locally.
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
from torch.utils.data import DataLoader, Dataset

# OpenOOD direct imports (no evaluation_api, evaluators, postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32


# ---------------------------------------------------------------------------
# Small test transform from openood/preprocessors/transform.py
# ---------------------------------------------------------------------------
class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_test_transform():
    """CIFAR-10 test transform: Resize(32), CenterCrop(32), ToTensor, Normalize."""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    return tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# EBO score: T * logsumexp(logits / T) with T=1
# ---------------------------------------------------------------------------
def ebo_score(logits, temperature=1.0):
    """Compute energy score for OOD detection."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation (percentage scale)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points (0-100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    # Sort by score descending (higher energy = more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    # True positive rate and false positive rate
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    if pos == 0 or neg == 0:
        return 50.0  # random chance

    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg

    # AUC = area under TPR as function of FPR
    auroc = np.trapz(tpr, fpr)
    return auroc * 100.0  # convert to percentage


# ---------------------------------------------------------------------------
# Evaluation function for one checkpoint
# ---------------------------------------------------------------------------
def evaluate_checkpoint(net, id_loader, ood_loaders, device='cpu'):
    """Compute EBO scores and AUROC for one model checkpoint."""
    net.eval()
    net.to(device)

    # Collect ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            logits = net(data)
            scores = ebo_score(logits).cpu().numpy()
            id_scores.append(scores)
    id_scores = np.concatenate(id_scores)

    # Collect OOD scores per dataset
    ood_results = {}
    for name, loader in ood_loaders.items():
        ood_scores = []
        with torch.no_grad():
            for batch in loader:
                data = batch['data'].to(device)
                logits = net(data)
                scores = ebo_score(logits).cpu().numpy()
                ood_scores.append(scores)
        ood_scores = np.concatenate(ood_scores)
        ood_results[name] = ood_scores

    # Compute AUROC for each OOD dataset
    aurocs = {}
    for name, ood_scores in ood_results.items():
        aurocs[name] = compute_auroc(id_scores, ood_scores)

    return aurocs, len(id_scores), {name: len(s) for name, s in ood_results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root (contains s0/, s1/, s2/)')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = 'cpu'

    # -----------------------------------------------------------------------
    # Data paths (OpenOOD default layout)
    # -----------------------------------------------------------------------
    data_dir = './data/images_classic'
    benchmark_dir = './data/benchmark_imglist/cifar10'

    # ID: CIFAR-10 test
    id_imglist = os.path.join(benchmark_dir, 'test_cifar10.txt')

    # Near-OOD: CIFAR-100 and TinyImageNet
    ood_configs = {
        'cifar100': os.path.join(benchmark_dir, 'test_cifar100.txt'),
        'tin': os.path.join(benchmark_dir, 'test_tin.txt'),
    }

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    transform = get_test_transform()

    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )

    ood_datasets = {}
    for name, imglist in ood_configs.items():
        ood_datasets[name] = ImglistDataset(
            name=f'{name}_test',
            imglist_pth=imglist,
            data_dir=data_dir,
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    id_loader = DataLoader(
        id_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    ood_loaders = {}
    for name, ds in ood_datasets.items():
        ood_loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )

    # -----------------------------------------------------------------------
    # Iterate over seeds s0, s1, s2
    # -----------------------------------------------------------------------
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        raise ValueError(f'No seed subdirectories found in {root}')

    run_metrics = {}
    dataset_counts = {}  # will be filled from first run

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        ckpt_path = os.path.join(seed_dir, 'best.ckpt')

        if not os.path.isfile(ckpt_path):
            print(f'Warning: checkpoint not found at {ckpt_path}', file=sys.stderr)
            continue

        # Load model
        net = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location=device)
        # Handle different checkpoint formats
        if 'state_dict' in state:
            net.load_state_dict(state['state_dict'])
        elif 'net' in state:
            net.load_state_dict(state['net'])
        else:
            net.load_state_dict(state)

        # Evaluate
        aurocs, n_id, n_ood = evaluate_checkpoint(net, id_loader, ood_loaders, device)

        run_metrics[seed_name] = aurocs
        if not dataset_counts:
            dataset_counts = {'cifar100': n_ood['cifar100'], 'tin': n_ood['tin']}

        print(f'Seed {seed_name}: {aurocs}', flush=True)

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    if not run_metrics:
        print('No checkpoints evaluated.', file=sys.stderr)
        sys.exit(1)

    # Per-run dataset mean
    run_means = []
    for seed_name, aurocs in run_metrics.items():
        ds_values = [aurocs[ds] for ds in ['cifar100', 'tin']]
        run_means.append(np.mean(ds_values))

    actual = float(np.mean(run_means))

    # Build output JSON
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
