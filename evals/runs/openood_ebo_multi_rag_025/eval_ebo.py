#!/usr/bin/env python3
"""Reproduce EBO near-OOD AUROC for CIFAR-10 with ResNet18_32x32.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Direct imports from openood modules (no evaluation_api, evaluators, postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.preprocessors.transform import normalization_dict, Convert, interpolation_modes


def build_test_transform():
    """Build the test transform for CIFAR-10 (base preprocessor test transform).

    This replicates TestStandardPreProcessor for CIFAR-10:
    - Convert to RGB
    - Resize to pre_size=32 (bilinear)
    - CenterCrop to image_size=32
    - ToTensor
    - Normalize with CIFAR-10 stats
    """
    mean = normalization_dict['cifar10'][0]
    std = normalization_dict['cifar10'][1]
    transform = tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(32, interpolation=interpolation_modes['bilinear']),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])
    return transform


def compute_ebo_scores(model, dataloader, temperature=1.0):
    """Compute EBO scores (negative energy) for all samples in dataloader.

    EBO score = -E(x) = T * log(sum(exp(f(x)/T)))
    Higher score -> more OOD-like.
    """
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch['data']
            labels = batch['label']
            logits = model(data)
            # Energy score: T * logsumexp(logits / T)
            energy = temperature * torch.logsumexp(logits / temperature, dim=1)
            # EBO uses energy as confidence (higher = more ID-like)
            scores = energy
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return all_scores, all_labels


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points.

    id_scores: scores for in-distribution samples (higher -> more ID-like)
    ood_scores: scores for out-of-distribution samples (higher -> more OOD-like)
    """
    # EBO energy score is higher for ID samples, so negate to make higher = more OOD-like
    scores = np.concatenate([-id_scores, -ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    auroc = roc_auc_score(labels, scores) * 100.0
    return auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root with s0/s1/s2 subfolders')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size

    # Validate root structure
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if len(subfolders) == 0:
        raise ValueError(f'No subfolders found in {root}')

    # Data paths (from official OpenOOD benchmark)
    data_dir = './data/images_classic/'
    imglist_base = './data/benchmark_imglist/cifar10/'

    # Build transform
    transform = build_test_transform()

    # ID dataset: CIFAR-10 test
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(imglist_base, 'test_cifar10.txt'),
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Near-OOD datasets
    ood_configs = [
        ('cifar100', os.path.join(imglist_base, 'test_cifar100.txt')),
        ('tin', os.path.join(imglist_base, 'test_tin.txt')),
    ]

    # Results storage
    run_metrics = {}
    dataset_counts = {}

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(checkpoint_path):
            print(f'Warning: checkpoint not found at {checkpoint_path}, skipping {run_name}', file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle potential 'net.' prefix in state dict
        if any(k.startswith('net.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.eval()

        # Compute ID scores once per run
        id_scores, _ = compute_ebo_scores(model, id_loader)
        id_count = len(id_scores)

        run_metrics[run_name] = {}
        for ood_name, ood_imglist in ood_configs:
            ood_dataset = ImglistDataset(
                name=f'{ood_name}_test',
                imglist_pth=ood_imglist,
                data_dir=data_dir,
                num_classes=10,
                preprocessor=transform,
                data_aux_preprocessor=transform,
            )
            ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            ood_scores, _ = compute_ebo_scores(model, ood_loader)
            ood_count = len(ood_scores)

            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ood_name] = auroc

            # Store dataset counts (same for all runs, but we record from first run)
            if ood_name not in dataset_counts:
                dataset_counts[ood_name] = ood_count

        # ID count is not included in the output datasets dict
        pass

    # Compute per-dataset mean across runs
    dataset_names = sorted(run_metrics[list(run_metrics.keys())[0]].keys())
    per_dataset_means = {}
    for dname in dataset_names:
        vals = [run_metrics[run][dname] for run in sorted(run_metrics.keys())]
        per_dataset_means[dname] = float(np.mean(vals))

    # Build output JSON with per-dataset metrics
    result = {
        'metric': 'near_ood_auroc',
        'actual': per_dataset_means,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'per_dataset_mean_across_runs',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
