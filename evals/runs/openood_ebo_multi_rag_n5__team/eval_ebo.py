#!/usr/bin/env python3
"""CPU-safe EBO evaluation for OpenOOD CIFAR-10 Near-OOD benchmark.

Reproduces the official AUROC using ResNet18_32x32 checkpoints s0/s1/s2,
EBO postprocessor (temperature=1.0), and the exact test transform from
openood/preprocessors/transform.py. Prints a single JSON REPRO_RESULT line.
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

# Direct imports from openood modules (no evaluation_api, evaluators, postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.preprocessors.transform import normalization_dict, interpolation_modes, Convert


def get_test_transform():
    """Return the exact test transform from TestStandardPreProcessor for CIFAR-10."""
    mean = normalization_dict['cifar10'][0]
    std = normalization_dict['cifar10'][1]
    return tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(32, interpolation=interpolation_modes['bilinear']),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])


def compute_ebo_scores(net, loader, device):
    """Compute EBO energy scores for all samples in loader.

    EBO score = temperature * logsumexp(logits / temperature, dim=1)
    with temperature=1.0. Higher energy -> more OOD-like.
    """
    net.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            data = batch['data'].to(device)
            logits = net(data)
            # EBO energy score
            energy = 1.0 * torch.logsumexp(logits / 1.0, dim=1)
            scores.append(energy.cpu())
    return torch.cat(scores).numpy()


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC treating higher scores as more OOD-like.

    Returns percentage (0-100).
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    # Higher score -> more OOD (label=1)
    auroc = roc_auc_score(labels, scores)
    return auroc * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/s1/s2 subfolders')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # Verify checkpoint structure
    subfolders = sorted([os.path.join(root, d) for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)) and d.startswith('s')])
    if len(subfolders) == 0:
        raise ValueError(f'No s0/s1/s2 subfolders found in {root}')

    # Data paths (relative to openood repo root, but we use absolute from --root parent)
    # Assume data is at ../data relative to root (standard OpenOOD layout)
    data_root = os.path.join(os.path.dirname(os.path.abspath(root)), 'data')

    # ID dataset: CIFAR-10 test
    id_imglist = os.path.join(data_root, 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    id_data_dir = os.path.join(data_root, 'images_classic')

    # Near-OOD datasets
    ood_configs = [
        ('cifar100', 'cifar100', 'test_cifar100.txt'),
        ('tin', 'tinyimagenet', 'test_tinyimagenet.txt'),
    ]

    transform = get_test_transform()

    # Pre-create ID dataset (shared across runs)
    id_dataset = ImglistDataset(
        name='cifar10',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)

    # Pre-create OOD datasets
    ood_datasets = {}
    ood_loaders = {}
    for ood_name, ood_subdir, ood_list in ood_configs:
        ood_imglist = os.path.join(data_root, 'benchmark_imglist', ood_subdir, ood_list)
        ood_data_dir = os.path.join(data_root, 'images_classic')
        ood_dataset = ImglistDataset(
            name=ood_name,
            imglist_pth=ood_imglist,
            data_dir=ood_data_dir,
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_datasets[ood_name] = ood_dataset
        ood_loaders[ood_name] = DataLoader(ood_dataset, batch_size=200, shuffle=False, num_workers=0)

    # Run evaluation for each seed
    run_metrics = {}
    for subfolder in subfolders:
        seed_name = os.path.basename(subfolder)
        ckpt_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        # Load model
        net = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        # Compute ID scores
        id_scores = compute_ebo_scores(net, id_loader, device)

        # Compute OOD scores and AUROC per dataset
        seed_metrics = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores = compute_ebo_scores(net, ood_loader, device)
            auroc = compute_auroc(id_scores, ood_scores)
            seed_metrics[ood_name] = round(auroc, 4)

        run_metrics[seed_name] = seed_metrics

    # Aggregate: dataset mean within each run, then mean of runs
    # First compute per-dataset means across runs
    dataset_means = {}
    for ood_name in ood_configs:
        ood_key = ood_name[0]
        values = [run_metrics[seed][ood_key] for seed in run_metrics]
        dataset_means[ood_key] = np.mean(values)

    # Then mean of dataset means = final metric
    actual = np.mean(list(dataset_means.values()))

    # Count evaluated samples (not runs)
    dataset_counts = {}
    for ood_name, ood_dataset in ood_datasets.items():
        dataset_counts[ood_name] = len(ood_dataset)
    dataset_counts['cifar100'] = len(ood_datasets['cifar100'])
    dataset_counts['tin'] = len(ood_datasets['tin'])

    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(float(actual), 4),
        'datasets': {
            'cifar100': len(ood_datasets['cifar100']),
            'tin': len(ood_datasets['tin']),
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
