#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC for CIFAR-10.

Reproduces the official OpenOOD evaluation for ResNet18_32x32 with EBO
postprocessor on CIFAR-10 ID vs CIFAR-100 and TinyImageNet Near-OOD.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default

Contract:
    - Uses only direct imports from openood.networks.resnet18_32x32 and
      openood.datasets.imglist_dataset.
    - Does NOT import openood.evaluation_api, openood.evaluators, or
      openood.postprocessors.
    - Implements EBO score and AUROC locally.
    - Prints exactly one strict-JSON REPRO_RESULT line.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

# Direct imports only — no evaluation_api, evaluators, or postprocessors
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# CIFAR-10 preprocessing (TestStandardPreProcessor equivalent)
# Verified against OpenOOD configs/preprocessors/base_preprocessor.yml
# and common CIFAR-10 normalization values.
# ---------------------------------------------------------------------------
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

test_transform = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# Dummy aux preprocessor (not used for EBO, but required by ImglistDataset)
aux_transform = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])


# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute EBO (energy) score: temperature * logsumexp(logits / temperature)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC (higher = better separation)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to results directory containing s0, s1, s2 subfolders')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Checkpoint subfolders
    seeds = ['s0', 's1', 's2']
    checkpoint_paths = []
    for seed in seeds:
        ckpt = os.path.join(root, seed, 'best.ckpt')
        if not os.path.isfile(ckpt):
            print(f'ERROR: Checkpoint not found: {ckpt}', file=sys.stderr)
            sys.exit(1)
        checkpoint_paths.append(ckpt)

    # OOD data lists (relative to OpenOOD repo root)
    # These paths are standard for CIFAR-10 Near-OOD in OpenOOD
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, 'data')

    # ID: CIFAR-10 test
    id_imglist = os.path.join(data_dir, 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    if not os.path.isfile(id_imglist):
        print(f'ERROR: ID imglist not found: {id_imglist}', file=sys.stderr)
        sys.exit(1)

    # Near-OOD datasets
    ood_configs = {
        'cifar100': {
            'imglist': os.path.join(data_dir, 'benchmark_imglist', 'cifar10', 'test_cifar100.txt'),
            'data_dir': os.path.join(data_dir, 'images_classic'),
        },
        'tin': {
            'imglist': os.path.join(data_dir, 'benchmark_imglist', 'cifar10', 'test_tin.txt'),
            'data_dir': os.path.join(data_dir, 'images_classic'),
        },
    }

    for ood_name, cfg in ood_configs.items():
        if not os.path.isfile(cfg['imglist']):
            print(f'ERROR: OOD imglist not found: {cfg["imglist"]}', file=sys.stderr)
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Per-seed evaluation
    # -----------------------------------------------------------------------
    run_metrics = {}  # {seed: {dataset: auroc}}
    dataset_counts = {}  # {dataset: count}

    for seed_idx, seed in enumerate(seeds):
        ckpt_path = checkpoint_paths[seed_idx]
        print(f'\n=== Evaluating seed {seed} ===', flush=True)

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location='cpu')
        # Handle possible 'state_dict' key
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        model.eval()

        # ID dataset
        id_dataset = ImglistDataset(
            name='cifar10_test',
            imglist_pth=id_imglist,
            data_dir=os.path.join(data_dir, 'images_classic'),
            num_classes=10,
            preprocessor=test_transform,
            data_aux_preprocessor=aux_transform,
        )
        id_loader = DataLoader(
            id_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Compute ID scores
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                logits = model(data)
                scores = ebo_score(logits, temperature=1.0)
                id_scores_list.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores_list)
        id_count = len(id_scores)

        # Evaluate each OOD dataset
        seed_metrics = {}
        for ood_name, cfg in ood_configs.items():
            ood_dataset = ImglistDataset(
                name=f'{ood_name}_test',
                imglist_pth=cfg['imglist'],
                data_dir=cfg['data_dir'],
                num_classes=10,
                preprocessor=test_transform,
                data_aux_preprocessor=aux_transform,
            )
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data']
                    logits = model(data)
                    scores = ebo_score(logits, temperature=1.0)
                    ood_scores_list.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores_list)

            # Store count (first seed only, all seeds have same dataset size)
            if seed == 's0':
                dataset_counts[ood_name] = len(ood_scores)

            # Compute AUROC
            auroc = compute_auroc(id_scores, ood_scores)
            # Convert to percentage points
            auroc_pct = auroc * 100.0
            seed_metrics[ood_name] = round(auroc_pct, 2)
            print(f'  {ood_name}: AUROC = {auroc_pct:.2f}%', flush=True)

        run_metrics[seed] = seed_metrics

    # Store ID count (same for all seeds)
    dataset_counts['cifar100'] = dataset_counts.get('cifar100', 0)
    dataset_counts['tin'] = dataset_counts.get('tin', 0)

    # -----------------------------------------------------------------------
    # Per-dataset mean and std across seeds
    # -----------------------------------------------------------------------
    per_dataset_metrics = {}
    for ood_name in ood_configs:
        values = [run_metrics[seed][ood_name] for seed in seeds]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1))
        per_dataset_metrics[ood_name] = {'mean': round(mean_val, 2), 'std': round(std_val, 2)}

    # -----------------------------------------------------------------------
    # Aggregation: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # Per-run dataset mean
    run_dataset_means = []
    for seed in seeds:
        metrics = run_metrics[seed]
        mean_val = np.mean(list(metrics.values()))
        run_dataset_means.append(mean_val)

    # Mean of runs
    actual = float(np.mean(run_dataset_means))
    actual = round(actual, 2)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
        'per_dataset_metrics': per_dataset_metrics,
    }

    print(f'\nREPRO_RESULT {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
