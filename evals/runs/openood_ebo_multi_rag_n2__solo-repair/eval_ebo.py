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
from torch.utils.data import DataLoader

# Import only the required modules from openood
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.base_preprocessor import BasePreprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 4

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official checkpoint paths (relative to root)
CHECKPOINT_REL = {
    's0': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt',
    's1': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt',
    's2': 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt',
}

# Near-OOD datasets: (name, imglist_rel, data_dir_rel)
NEAR_OOD_DATASETS = [
    ('cifar100', 'data/benchmark_imglist/cifar100/test_cifar10.txt', 'data/images_classic'),
    ('tin', 'data/benchmark_imglist/cifar100/test_tin.txt', 'data/images_classic'),
]

# ---------------------------------------------------------------------------
# Transform pipeline (exactly as in TestStandardPreProcessor)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the test transform for CIFAR-10 (Resize, CenterCrop, ToTensor, Normalize)."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Compute Energy-Based OOD scores: -logsumexp(logits / T)."""
    scores = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return scores.cpu().numpy()

# ---------------------------------------------------------------------------
# AUROC calculation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points. Higher score = more OOD-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # Sort by score descending (higher score -> more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    # True positive rate and false positive rate
    tpr = np.cumsum(labels_sorted == 1) / np.sum(labels_sorted == 1)
    fpr = np.cumsum(labels_sorted == 0) / np.sum(labels_sorted == 0)
    # AUC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)  # percentage

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # Build test transform
    transform = get_test_transform()

    # Create a minimal preprocessor that wraps our transform
    class SimplePreprocessor(BasePreprocessor):
        def __init__(self, transform):
            self.transform = transform
        def setup(self, **kwargs):
            pass
        def __call__(self, image):
            return self.transform(image)

    preprocessor = SimplePreprocessor(transform)

    # Load ID dataset (CIFAR-10 test) once for all runs
    id_imglist = os.path.join(root, 'data/benchmark_imglist/cifar10/test_cifar10.txt')
    id_data_dir = os.path.join(root, 'data/images_classic')
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=preprocessor,
        data_aux_preprocessor=preprocessor
    )
    id_loader = DataLoader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Prepare OOD datasets
    ood_datasets = {}
    for name, imglist_rel, data_dir_rel in NEAR_OOD_DATASETS:
        imglist_pth = os.path.join(root, imglist_rel)
        data_dir = os.path.join(root, data_dir_rel)
        ds = ImglistDataset(
            name=name,
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=NUM_CLASSES,
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor
        )
        ood_datasets[name] = ds

    # Results storage
    run_metrics = {}
    dataset_counts = {}

    for run_key in ['s0', 's1', 's2']:
        # Load checkpoint
        ckpt_path = os.path.join(root, CHECKPOINT_REL[run_key])
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location=device)
        # Handle possible 'state_dict' key
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        # Compute ID scores
        id_scores_list = []
        with torch.no_grad():
            for images, _ in id_loader:
                images = images.to(device)
                logits = model(images)
                scores = compute_ebo_scores(logits)
                id_scores_list.append(scores)
        id_scores = np.concatenate(id_scores_list)

        # Compute OOD scores for each dataset
        ood_results = {}
        for ood_name, ood_ds in ood_datasets.items():
            ood_loader = DataLoader(
                ood_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=False,
            )
            ood_scores_list = []
            with torch.no_grad():
                for images, _ in ood_loader:
                    images = images.to(device)
                    logits = model(images)
                    scores = compute_ebo_scores(logits)
                    ood_scores_list.append(scores)
            ood_scores = np.concatenate(ood_scores_list)
            auroc = compute_auroc(id_scores, ood_scores)
            ood_results[ood_name] = auroc
            # Store dataset count (number of OOD samples)
            dataset_counts[ood_name] = len(ood_ds)

        run_metrics[run_key] = ood_results

    # Store ID dataset count (same for all runs)
    dataset_counts['cifar100'] = len(ood_datasets['cifar100'])
    dataset_counts['tin'] = len(ood_datasets['tin'])

    # Compute aggregation: dataset mean within each run, then mean of runs
    # For each run, compute mean AUROC across datasets
    run_means = []
    for run_key in ['s0', 's1', 's2']:
        vals = list(run_metrics[run_key].values())
        run_mean = float(np.mean(vals))
        run_means.append(run_mean)
    actual = float(np.mean(run_means))

    # Build output JSON
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one line with REPRO_RESULT prefix
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
