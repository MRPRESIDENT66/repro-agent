#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Direct imports from openood (no evaluators / postprocessors / evaluation_api)
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants – official CIFAR-10 test transform (from openood/preprocessors/transform.py
# and base_preprocessor.py)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# The test transform used by TestStandardPreProcessor for CIFAR-10:
#   Convert('RGB') -> Resize(32, interpolation=bilinear) -> CenterCrop(32) -> ToTensor -> Normalize
# We hardcode the interpolation mode (bilinear) and sizes (32).
TEST_TRANSFORM = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Checkpoint paths (relative to root)
# ---------------------------------------------------------------------------
CHECKPOINT_RELPATHS = {
    's0': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    's1': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt',
    's2': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt',
}

# ---------------------------------------------------------------------------
# Benchmark image list paths (relative to root)
# ---------------------------------------------------------------------------
ID_LIST = 'data/benchmark_imglist/cifar10/test_cifar10.txt'
OOD_LISTS = {
    'cifar100': 'data/benchmark_imglist/cifar10/test_cifar100.txt',
    'tin': 'data/benchmark_imglist/cifar10/test_tinyimagenet.txt',
}

# ---------------------------------------------------------------------------
# Helper: wrap ImglistDataset with our test transform
# ---------------------------------------------------------------------------
class TransformedImglistDataset(Dataset):
    """Apply a fixed transform to ImglistDataset items (image, label)."""
    def __init__(self, root, imglist_pth, transform):
        self.base = ImglistDataset(root=root, imglist_pth=imglist_pth)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # img is a PIL image
        img = self.transform(img)
        return img, label

# ---------------------------------------------------------------------------
# EBO score: max softmax logit (energy-based)
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor) -> torch.Tensor:
    """Return EBO score = -log(sum(exp(logits))) per sample."""
    # logsumexp over class dimension
    return -torch.logsumexp(logits, dim=1)

# ---------------------------------------------------------------------------
# AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points. Higher score = more OOD-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # Sort by score descending (higher score -> more OOD)
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
# Main evaluation routine
# ---------------------------------------------------------------------------
def evaluate(root: str):
    device = torch.device('cpu')

    # Load ID dataset (CIFAR-10 test)
    id_dataset = TransformedImglistDataset(
        root=root,
        imglist_pth=os.path.join(root, ID_LIST),
        transform=TEST_TRANSFORM,
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)

    # Load OOD datasets
    ood_datasets = {}
    for name, relpath in OOD_LISTS.items():
        ds = TransformedImglistDataset(
            root=root,
            imglist_pth=os.path.join(root, relpath),
            transform=TEST_TRANSFORM,
        )
        ood_datasets[name] = DataLoader(ds, batch_size=200, shuffle=False, num_workers=0)

    # Results storage
    run_metrics = {}  # run -> dataset -> auroc
    dataset_counts = {}  # dataset -> number of samples (from ID or OOD)

    for run_name, rel_ckpt in CHECKPOINT_RELPATHS.items():
        # Load model
        model = ResNet18_32x32(num_classes=10)
        ckpt_path = os.path.join(root, rel_ckpt)
        state = torch.load(ckpt_path, map_location=device)
        # The checkpoint may contain 'state_dict' key or be the state_dict directly
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)

        # Compute ID scores
        id_scores_list = []
        with torch.no_grad():
            for images, _ in id_loader:
                images = images.to(device)
                logits = model(images)
                scores = ebo_score(logits)  # higher = more OOD
                id_scores_list.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores_list)
        # Store count (ID samples are the same for all runs)
        if 'cifar10' not in dataset_counts:
            dataset_counts['cifar10'] = len(id_scores)

        # Compute OOD scores per dataset
        run_metrics[run_name] = {}
        for ood_name, ood_loader in ood_datasets.items():
            ood_scores_list = []
            with torch.no_grad():
                for images, _ in ood_loader:
                    images = images.to(device)
                    logits = model(images)
                    scores = ebo_score(logits)
                    ood_scores_list.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores_list)
            # Store count (first run)
            if ood_name not in dataset_counts:
                dataset_counts[ood_name] = len(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ood_name] = auroc

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # For each run, compute mean over datasets (cifar100, tin)
    run_avgs = []
    for run_name in ['s0', 's1', 's2']:
        vals = [run_metrics[run_name][d] for d in ['cifar100', 'tin']]
        run_avgs.append(np.mean(vals))
    actual = float(np.mean(run_avgs))

    # Build output JSON
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': {
            'cifar100': dataset_counts.get('cifar100', 0),
            'tin': dataset_counts.get('tin', 0),
        },
        'run_metrics': {
            run: {
                'cifar100': run_metrics[run]['cifar100'],
                'tin': run_metrics[run]['tin'],
            }
            for run in ['s0', 's1', 's2']
        },
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.',
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()
    evaluate(args.root)


if __name__ == '__main__':
    main()
