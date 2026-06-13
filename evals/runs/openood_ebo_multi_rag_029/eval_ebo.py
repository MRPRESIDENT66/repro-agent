#!/usr/bin/env python3
"""Reproduce EBO Near-OOD AUROC for CIFAR-10 using official checkpoints."""

import argparse
import glob
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
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants from openood/preprocessors/transform.py
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ---------------------------------------------------------------------------
# Small test transform (directly from TestStandardPreProcessor logic)
# ---------------------------------------------------------------------------
def build_test_transform():
    """Build the CIFAR-10 test transform as in TestStandardPreProcessor."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy score E(x) = T * logsumexp(f(x)/T)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points. Higher energy -> more OOD-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # For energy: higher score = more OOD, so we want positive class to be OOD
    auroc = roc_auc_score(labels, scores)
    return auroc * 100.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_run(net: torch.nn.Module,
                 id_loader: DataLoader,
                 ood_loader: DataLoader,
                 device: torch.device) -> float:
    """Evaluate a single run and return AUROC for one OOD dataset."""
    net.eval()
    id_energies = []
    ood_energies = []

    for batch in id_loader:
        data = batch['data'].to(device)
        logits = net(data)
        energies = energy_score(logits)
        id_energies.append(energies.detach().cpu().numpy())

    for batch in ood_loader:
        data = batch['data'].to(device)
        logits = net(data)
        energies = energy_score(logits)
        ood_energies.append(energies.detach().cpu().numpy())

    id_scores = np.concatenate(id_energies)
    ood_scores = np.concatenate(ood_energies)
    return compute_auroc(id_scores, ood_scores)


def main():
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC Reproduction')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    parser.add_argument('--data-root', type=str, default='./data/images_classic/',
                        help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    root = args.root
    data_root = args.data_root
    batch_size = args.batch_size
    device = torch.device('cpu')

    # -----------------------------------------------------------------------
    # Dataset paths (official benchmark lists)
    # -----------------------------------------------------------------------
    imglist_dir = './data/benchmark_imglist/cifar10'
    id_imglist = os.path.join(imglist_dir, 'test_cifar10.txt')
    ood_imglists = {
        'cifar100': './data/benchmark_imglist/cifar10/test_cifar100.txt',
        'tin': './data/benchmark_imglist/cifar10/test_tin.txt',
    }

    # -----------------------------------------------------------------------
    # Build transform and datasets
    # -----------------------------------------------------------------------
    transform = build_test_transform()

    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_root,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ood_loaders = {}
    for name, imglist_pth in ood_imglists.items():
        dataset = ImglistDataset(
            name=f'{name}_test',
            imglist_pth=imglist_pth,
            data_dir=data_root,
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Find checkpoint subfolders
    # -----------------------------------------------------------------------
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if not subfolders:
        raise ValueError(f'No subfolders (s0, s1, s2) found in {root}')

    # -----------------------------------------------------------------------
    # Evaluate each run
    # -----------------------------------------------------------------------
    run_metrics = {}
    dataset_counts = {}

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        ckpt_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        # Load model
        net = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)

        # Evaluate on each OOD dataset
        run_metrics[run_name] = {}
        for ood_name, ood_loader in ood_loaders.items():
            auroc = evaluate_run(net, id_loader, ood_loader, device)
            run_metrics[run_name][ood_name] = round(auroc, 2)

        # Record dataset counts (number of ID samples)
        if not dataset_counts:
            dataset_counts['cifar100'] = len(ood_loaders['cifar100'].dataset)
            dataset_counts['tin'] = len(ood_loaders['tin'].dataset)

    # -----------------------------------------------------------------------
    # Aggregation: per-dataset mean across runs (official OpenOOD method)
    # -----------------------------------------------------------------------
    # For each dataset, compute mean AUROC across runs
    dataset_aurocs = {}
    for ood_name in ood_loaders.keys():
        aurocs = [run_metrics[run_name][ood_name] for run_name in run_metrics]
        dataset_aurocs[ood_name] = round(float(np.mean(aurocs)), 2)

    # Compute overall actual as mean of per-dataset means
    actual = float(np.mean(list(dataset_aurocs.values())))

    # -----------------------------------------------------------------------
    # Output strict JSON
    # -----------------------------------------------------------------------
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 2),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
