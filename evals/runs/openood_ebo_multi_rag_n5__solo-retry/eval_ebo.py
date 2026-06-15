#!/usr/bin/env python3
"""Reproduce OpenOOD EBO Near-OOD AUROC for CIFAR-10 (CPU)."""

import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from torch.utils.data import DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Direct imports from openood (no evaluators / postprocessors / evaluation_api)
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# 2. Transform pipeline (retrieved from openood/preprocessors/transform.py
#    and openood/preprocessors/test_preprocessor.py)
# ---------------------------------------------------------------------------
# CIFAR-10 normalization constants from normalization_dict
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# The test transform used by TestStandardPreProcessor for CIFAR-10:
# Convert('RGB'), Resize(32, interpolation=bilinear), CenterCrop(32),
# ToTensor(), Normalize(mean, std)
# pre_size=32, image_size=32, interpolation='bilinear' from config.
test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# 3. Helper: build ImglistDataset with the transform as preprocessor
# ---------------------------------------------------------------------------
def build_dataset(imglist_pth, data_dir):
    """Return an ImglistDataset with the test transform."""
    # ImglistDataset expects preprocessor and data_aux_preprocessor.
    # We pass the same transform for both (no auxiliary augmentation).
    dataset = ImglistDataset(
        name='cifar10',
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )
    return dataset

# ---------------------------------------------------------------------------
# 4. EBO score function (energy-based)
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute -logsumexp(logits / T) * T as the energy score."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# 5. AUROC computation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points.
    Higher score -> more OOD-like (energy score: lower energy = more ID).
    We treat lower energy as ID, higher energy as OOD.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # Sort by score descending (higher score -> more OOD)
    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0  # random

    tpr = 0.0
    fpr = 0.0
    prev_score = None
    auroc = 0.0
    fpr_list = [0.0]
    tpr_list = [0.0]

    for i in range(len(scores)):
        if prev_score is not None and scores[sorted_idx[i]] != prev_score:
            # trapezoidal integration
            auroc += np.trapz([tpr, tpr], [fpr, fpr])  # vertical step
            # Actually we need to accumulate area under TPR as function of FPR.
            # Simpler: use the standard formula.
            pass
        prev_score = scores[sorted_idx[i]]
        if sorted_labels[i] == 1:
            tpr += 1.0 / pos
        else:
            fpr += 1.0 / neg
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # Use trapezoidal integration
    auroc = np.trapz(tpr_list, fpr_list)
    return auroc * 100.0  # percentage

# ---------------------------------------------------------------------------
# 6. Main evaluation routine
# ---------------------------------------------------------------------------
def evaluate_checkpoint(ckpt_path: str, id_loader: DataLoader,
                        ood_loaders: dict) -> dict:
    """Load checkpoint, compute EBO scores, return per-OOD AUROC dict."""
    device = torch.device('cpu')
    model = ResNet18_32x32(num_classes=10)
    state = torch.load(ckpt_path, map_location=device)
    # The checkpoint may contain 'state_dict' or be the state_dict itself.
    if 'state_dict' in state:
        state = state['state_dict']
    # Remove 'module.' prefix if present (DataParallel wrapping)
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    model.eval()

    # Collect ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            images = batch['data'].to(device)
            logits = model(images)
            scores = energy_score(logits).cpu().numpy()
            id_scores.append(scores)
    id_scores = np.concatenate(id_scores)

    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = energy_score(logits).cpu().numpy()
                ood_scores.append(scores)
        ood_scores = np.concatenate(ood_scores)
        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc
    return results

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace',
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    root = args.root
    data_dir = os.path.join(root, 'data', 'images_classic')
    benchmark_dir = os.path.join(root, 'data', 'benchmark_imglist', 'cifar10')
    checkpoint_dir = os.path.join(root, 'results',
                                  'cifar10_resnet18_32x32_base_e100_lr0.1_default')

    # ID dataset: CIFAR-10 test
    id_imglist = os.path.join(benchmark_dir, 'test_cifar10.txt')
    id_dataset = build_dataset(id_imglist, data_dir)
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)

    # OOD datasets
    ood_configs = {
        'cifar100': {
            'imglist': os.path.join(root, 'data', 'benchmark_imglist', 'cifar100',
                                    'test_cifar100.txt'),
            'data_dir': os.path.join(root, 'data', 'images_classic'),
        },
        'tin': {
            'imglist': os.path.join(root, 'data', 'benchmark_imglist', 'tin',
                                    'test_tin.txt'),
            'data_dir': os.path.join(root, 'data', 'images_classic'),
        },
    }

    ood_loaders = {}
    ood_counts = {}
    for name, cfg in ood_configs.items():
        ds = build_dataset(cfg['imglist'], cfg['data_dir'])
        ood_loaders[name] = DataLoader(ds, batch_size=200, shuffle=False, num_workers=0)
        ood_counts[name] = len(ds)

    # Checkpoints s0, s1, s2
    seeds = ['s0', 's1', 's2']
    run_metrics = {}
    for seed in seeds:
        ckpt_path = os.path.join(checkpoint_dir, seed, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            # fallback to epoch_96.ckpt or similar
            ckpt_path = os.path.join(checkpoint_dir, seed, 'epoch_96.ckpt')
        if not os.path.exists(ckpt_path):
            # try any .ckpt
            ckpt_dir = os.path.join(checkpoint_dir, seed)
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpts:
                ckpt_path = os.path.join(ckpt_dir, ckpts[0])
            else:
                raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
        results = evaluate_checkpoint(ckpt_path, id_loader, ood_loaders)
        run_metrics[seed] = results

    # Compute dataset mean per run, then mean of runs
    dataset_names = list(ood_configs.keys())
    # For each dataset, compute mean across runs
    dataset_means = {}
    for ds_name in dataset_names:
        vals = [run_metrics[seed][ds_name] for seed in seeds]
        dataset_means[ds_name] = np.mean(vals)

    # Then mean of dataset means
    actual = np.mean(list(dataset_means.values()))

    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': ood_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print('REPRO_RESULT ' + json.dumps(output))

if __name__ == '__main__':
    main()
