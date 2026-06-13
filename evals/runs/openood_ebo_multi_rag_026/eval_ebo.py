#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO evaluation for CIFAR-10 Near-OOD.

Reproduces official OpenOOD EBO AUROC on CIFAR-100 and TinyImageNet
using s0/s1/s2 ResNet18_32x32 checkpoints. Prints strict JSON REPRO_RESULT.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.preprocessors.transform import Convert, normalization_dict, interpolation_modes


def build_test_transform():
    """Build the test transform matching TestStandardPreProcessor for CIFAR-10."""
    mean = normalization_dict['cifar10'][0]
    std = normalization_dict['cifar10'][1]
    return tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(32, interpolation=interpolation_modes['bilinear']),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])


def load_dataset(imglist_name, data_root, imglist_root):
    """Load a dataset using ImglistDataset with the test transform."""
    transform = build_test_transform()
    # For test, data_aux_preprocessor is same as preprocessor (no extra aug)
    dataset = ImglistDataset(
        name=imglist_name,
        imglist_pth=os.path.join(imglist_root, f'{imglist_name}.txt'),
        data_dir=data_root,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    return dataset


def compute_ebo_scores(logits, temperature=1.0):
    """Compute EBO scores: temperature * logsumexp(logits / temperature, dim=1)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC: higher score = ID, lower = OOD."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # EBO scores: higher = more OOD, so negate for AUROC (higher = ID)
    return roc_auc_score(labels, -scores)


def evaluate_checkpoint(ckpt_path, id_loader, ood_loaders, device):
    """Evaluate a single checkpoint on ID and OOD datasets, return AUROCs."""
    # Load model
    model = ResNet18_32x32(num_classes=10)
    state = torch.load(ckpt_path, map_location=device)
    # Handle possible 'state_dict' key
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Collect ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            logits = model(data)
            scores = compute_ebo_scores(logits)
            id_scores.append(scores.cpu().numpy())
    id_scores = np.concatenate(id_scores)

    # Collect OOD scores per dataset
    aurocs = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                logits = model(data)
                scores = compute_ebo_scores(logits)
                ood_scores.append(scores.cpu().numpy())
        ood_scores = np.concatenate(ood_scores)
        aurocs[ood_name] = compute_auroc(id_scores, ood_scores)

    return aurocs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root (contains s0/, s1/, s2/)')
    parser.add_argument('--data-root', type=str, default='./data/images_classic/',
                        help='Path to image data directory')
    parser.add_argument('--imglist-root', type=str,
                        default='./data/benchmark_imglist/cifar10/',
                        help='Path to benchmark imglist directory')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load datasets
    print('Loading datasets...', file=sys.stderr)
    id_dataset = load_dataset('test_cifar10', args.data_root, args.imglist_root)
    ood_datasets = {
        'cifar100': load_dataset('test_cifar100', args.data_root, args.imglist_root),
        'tin': load_dataset('test_tin', args.data_root, args.imglist_root),
    }

    id_loader = DataLoader(id_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    ood_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)
        for name, ds in ood_datasets.items()
    }

    # Find checkpoint subdirectories
    ckpt_dirs = sorted([
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d)) and d.startswith('s')
    ])
    if not ckpt_dirs:
        raise ValueError(f'No s0/s1/s2 subdirectories found in {args.root}')

    # Evaluate each checkpoint
    run_metrics = {}
    for run_name in ckpt_dirs:
        ckpt_path = os.path.join(args.root, run_name, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        print(f'Evaluating {run_name}...', file=sys.stderr)
        aurocs = evaluate_checkpoint(ckpt_path, id_loader, ood_loaders, device)
        # Convert to percentage
        run_metrics[run_name] = {
            name: round(val * 100, 4) for name, val in aurocs.items()
        }

    # Compute dataset counts
    dataset_counts = {
        'cifar100': len(ood_datasets['cifar100']),
        'tin': len(ood_datasets['tin']),
    }

    # Compute aggregation: dataset mean within each run, then mean of runs
    run_means = []
    for run_name in ckpt_dirs:
        vals = list(run_metrics[run_name].values())
        run_means.append(np.mean(vals))
    actual = float(np.mean(run_means))

    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 4),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print strict JSON
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
