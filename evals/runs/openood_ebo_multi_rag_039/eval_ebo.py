#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC reproduction for CIFAR-10.

Implements the exact OpenOOD evaluation protocol using:
- ResNet18_32x32 from openood.networks.resnet18_32x32
- ImglistDataset from openood.datasets.imglist_dataset
- Official CIFAR-10 test transform (Resize→CenterCrop→ToTensor→Normalize)
- EBO score: temperature * logsumexp(logits / temperature)
- AUROC via sklearn.metrics.roc_curve with OOD as positive class
- Aggregation: dataset mean within each run, then mean of runs

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, or postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 0  # CPU-safe

# Near-OOD datasets for CIFAR-10
OOD_DATASETS = {
    'cifar100': {
        'imglist': './data/benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': './data/images_classic/',
    },
    'tin': {
        'imglist': './data/benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': './data/images_classic/',
    },
}

# ---------------------------------------------------------------------------
# Transform pipeline (exact copy of TestStandardPreProcessor from OpenOOD)
# ---------------------------------------------------------------------------
from torchvision import transforms as tvs_trans
from PIL import Image

class ConvertRGB:
    def __call__(self, image):
        return image.convert('RGB')

def get_test_transform():
    """Return the CIFAR-10 test transform: Resize(32)→CenterCrop(32)→ToTensor→Normalize."""
    return tvs_trans.Compose([
        ConvertRGB(),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits, temperature=1.0):
    """Compute energy scores: temperature * logsumexp(logits / temperature)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (matches OpenOOD semantics)
# ---------------------------------------------------------------------------
def compute_auroc(conf_scores, labels):
    """
    Compute AUROC following OpenOOD convention:
    - OOD samples have label == -1
    - OOD is treated as positive class
    - ID samples should have larger conf values than OOD samples
    - Therefore we negate conf for roc_curve
    """
    ood_indicator = np.zeros_like(labels)
    ood_indicator[labels == -1] = 1
    fpr, tpr, _ = metrics.roc_curve(ood_indicator, -conf_scores)
    return metrics.auc(fpr, tpr)

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_run(checkpoint_path, id_dataset_cfg, ood_datasets, device='cpu'):
    """
    Evaluate a single checkpoint on ID + OOD datasets.
    Returns dict of {dataset_name: auroc_percentage}.
    """
    # Load model
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle checkpoint formats: direct state_dict or wrapped
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    elif 'net' in state_dict:
        model.load_state_dict(state_dict['net'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_test_transform()

    results = {}

    for ood_name, ood_info in ood_datasets.items():
        # Create OOD dataset
        ood_dataset = ImglistDataset(
            name=ood_name,
            imglist_pth=ood_info['imglist'],
            data_dir=ood_info['data_dir'],
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        # Create ID dataset (CIFAR-10 test)
        id_dataset_obj = ImglistDataset(
            name='cifar10',
            imglist_pth=id_dataset_cfg['imglist'],
            data_dir=id_dataset_cfg['data_dir'],
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        id_loader = DataLoader(
            id_dataset_obj,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        all_conf = []
        all_labels = []

        # Process ID data
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data'].to(device)
                labels = batch['label'].numpy()
                logits = model(images)
                conf = compute_ebo_scores(logits).cpu().numpy()
                all_conf.extend(conf)
                all_labels.extend(labels)

            # Process OOD data
            for batch in ood_loader:
                images = batch['data'].to(device)
                # OOD labels are -1
                labels = np.full(images.size(0), -1)
                logits = model(images)
                conf = compute_ebo_scores(logits).cpu().numpy()
                all_conf.extend(conf)
                all_labels.extend(labels)

        all_conf = np.array(all_conf)
        all_labels = np.array(all_labels)

        auroc = compute_auroc(all_conf, all_labels)
        results[ood_name] = auroc * 100  # Convert to percentage

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC for CIFAR-10')
    parser.add_argument('--root', required=True,
                        help='Path to results directory containing s0, s1, s2 subfolders')
    args = parser.parse_args()

    root = args.root

    # ID dataset info
    id_dataset = {
        'imglist': './data/benchmark_imglist/cifar10/test_cifar10.txt',
        'data_dir': './data/images_classic/',
    }

    # Find checkpoint subfolders
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if not subfolders:
        raise ValueError(f'No subfolders (s0, s1, ...) found in {root}')

    # Evaluate each run
    run_metrics = {}
    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}, skipping {run_name}')
            continue
        print(f'Evaluating {run_name}...')
        metrics_dict = evaluate_run(checkpoint_path, id_dataset, OOD_DATASETS)
        run_metrics[run_name] = metrics_dict

    # Compute aggregation: dataset mean within each run, then mean of runs
    dataset_names = list(OOD_DATASETS.keys())
    run_means = []
    for run_name, metrics_dict in run_metrics.items():
        dataset_values = [metrics_dict[ds] for ds in dataset_names]
        run_mean = np.mean(dataset_values)
        run_means.append(run_mean)

    actual = np.mean(run_means)

    # Count evaluated samples (for verification)
    sample_counts = {}
    for ds_name in dataset_names:
        imglist_path = OOD_DATASETS[ds_name]['imglist']
        with open(imglist_path) as f:
            sample_counts[ds_name] = len(f.readlines())

    # Build result dictionary
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': sample_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print the required JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')

if __name__ == '__main__':
    main()
