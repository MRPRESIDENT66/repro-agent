#!/usr/bin/env python3
"""eval_ebo.py — CPU-safe EBO evaluation for CIFAR-10 near-OOD detection.

Executes the exact OpenOOD evaluation protocol using:
- ResNet18_32x32 with official s0/s1/s2 checkpoints
- ImglistDataset for benchmark image lists
- Local EBO score computation (temperature=1.0)
- sklearn AUROC with OOD-as-positive convention
- Dataset-then-run mean aggregation
- Strict JSON REPRO_RESULT output
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

# Direct imports from OpenOOD modules (no evaluation_api, evaluators, or postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

ID_NAME = 'cifar10'
ID_NUM_CLASSES = 10
ID_IMG_LIST = 'data/benchmark_imglist/cifar10/test_cifar10.txt'
ID_DATA_DIR = 'data/images_classic/'

NEAR_OOD_DATASETS = {
    'cifar100': {
        'imglist': 'data/benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': 'data/images_classic/',
    },
    'tin': {
        'imglist': 'data/benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': 'data/images_classic/',
    },
}

BATCH_SIZE = 128
TEMPERATURE = 1.0

# ---------------------------------------------------------------------------
# Transform (directly from openood/preprocessors/transform.py semantics)
# ---------------------------------------------------------------------------
def get_test_transform():
    """CIFAR-10 test transform: ToTensor + Normalize (no resize for 32x32)."""
    return tvs_trans.Compose([
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits: torch.Tensor, temperature: float = TEMPERATURE) -> np.ndarray:
    """Compute EBO (energy) scores: temperature * logsumexp(logits / temperature)."""
    scores = temperature * torch.logsumexp(logits / temperature, dim=1)
    return scores.cpu().numpy()

# ---------------------------------------------------------------------------
# AUROC computation (OOD as positive class)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC with OOD as positive class.

    EBO produces higher scores for ID, so we negate scores before ROC analysis.
    Returns percentage (0-100).
    """
    scores = np.concatenate([-id_scores, -ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    auroc = roc_auc_score(labels, scores)
    return auroc * 100.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(imglist_path: str, data_dir: str, transform) -> ImglistDataset:
    """Load an ImglistDataset with the given transform."""
    return ImglistDataset(
        name='eval',
        imglist_pth=imglist_path,
        data_dir=data_dir,
        num_classes=ID_NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='EBO evaluation for CIFAR-10')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/s1/s2 subfolders with best.ckpt')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f'Error: root directory {root} does not exist', file=sys.stderr)
        sys.exit(1)

    # Find checkpoint subfolders
    subfolders = sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith('s')
    ])
    if not subfolders:
        print(f'Error: no s0/s1/s2 subfolders found in {root}', file=sys.stderr)
        sys.exit(1)

    # Build transform
    transform = get_test_transform()

    # Load ID dataset once (same for all runs)
    id_dataset = load_dataset(ID_IMG_LIST, ID_DATA_DIR, transform)
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load OOD datasets once
    ood_datasets = {}
    ood_loaders = {}
    for name, info in NEAR_OOD_DATASETS.items():
        ds = load_dataset(info['imglist'], info['data_dir'], transform)
        ood_datasets[name] = ds
        ood_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Store per-run, per-dataset AUROC
    run_metrics = {}  # {run_name: {dataset_name: auroc}}

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}, skipping', file=sys.stderr)
            continue

        # Load model
        net = ResNet18_32x32(num_classes=ID_NUM_CLASSES)
        state = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(state)
        net.eval()

        # Inference on ID
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                logits = net(data)
                scores = compute_ebo_scores(logits, TEMPERATURE)
                id_scores_list.append(scores)
        id_scores = np.concatenate(id_scores_list)

        # Inference on each OOD dataset
        run_metrics[run_name] = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data']
                    logits = net(data)
                    scores = compute_ebo_scores(logits, TEMPERATURE)
                    ood_scores_list.append(scores)
            ood_scores = np.concatenate(ood_scores_list)
            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ood_name] = auroc

    # -----------------------------------------------------------------------
    # Aggregation: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    if not run_metrics:
        print('Error: no runs produced metrics', file=sys.stderr)
        sys.exit(1)

    # Per-run dataset means
    run_dataset_means = {}
    for run_name, metrics in run_metrics.items():
        run_dataset_means[run_name] = np.mean(list(metrics.values()))

    # Overall mean across runs
    actual = float(np.mean(list(run_dataset_means.values())))

    # Dataset counts (evaluated samples per dataset)
    dataset_counts = {}
    for name, ds in ood_datasets.items():
        dataset_counts[name] = len(ds)

    # Build output JSON
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one strict-JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
