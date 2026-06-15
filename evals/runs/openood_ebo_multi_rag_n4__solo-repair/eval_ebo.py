#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Direct imports from openood modules (no evaluation_api, evaluators, postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace')
    return parser.parse_args()


def get_test_transform():
    """Return the exact test transform from openood/preprocessors/transform.py
    for CIFAR-10: Resize(32), CenterCrop(32), ToTensor, Normalize(cifar10).
    The official CIFAR-10 normalization uses std=[0.247, 0.2435, 0.2616]."""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.2435, 0.2616]),
    ])


def load_model(checkpoint_path, device):
    """Load ResNet18_32x32 with 10 classes from checkpoint."""
    model = ResNet18_32x32(num_classes=10)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle both direct state_dict and wrapped checkpoint
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'net' in state:
        state = state['net']
    # Remove 'module.' prefix if present
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_ebo_scores(model, loader, temperature=1.0, device='cpu'):
    """Compute EBO energy scores for all samples in loader."""
    scores = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            data = batch['data'].to(device)
            label = batch['label']
            logits = model(data)
            # EBO score: temperature * logsumexp(logits / temperature)
            energy = temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.append(energy.cpu())
            labels.append(label)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return scores, labels


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC given ID and OOD energy scores.
    Higher energy -> more OOD-like. We treat ID as positive class."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending (higher energy = more OOD)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    # Compute TPR and FPR
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    tpr = np.cumsum(sorted_labels == 1) / pos_count
    fpr = np.cumsum(sorted_labels == 0) / neg_count
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return auroc * 100  # Return in percentage points


def main():
    args = parse_args()
    root = args.root
    device = 'cpu'

    # Paths
    data_dir = os.path.join(root, 'data', 'images_classic')
    imglist_dir = os.path.join(root, 'data', 'benchmark_imglist', 'cifar10')
    ood_imglist_dir = os.path.join(root, 'data', 'benchmark_imglist')
    checkpoint_dir = os.path.join(root, 'results',
                                  'cifar10_resnet18_32x32_base_e100_lr0.1_default')
    # If root already points to the checkpoint directory, adjust
    if os.path.basename(root) == 'cifar10_resnet18_32x32_base_e100_lr0.1_default':
        checkpoint_dir = root
        data_dir = os.path.join(os.path.dirname(root), 'data', 'images_classic')
        imglist_dir = os.path.join(os.path.dirname(root), 'data', 'benchmark_imglist', 'cifar10')
        ood_imglist_dir = os.path.join(os.path.dirname(root), 'data', 'benchmark_imglist')
        # When root is the checkpoint directory, checkpoints are directly in s0/, s1/, s2/
        checkpoint_paths = {
            's0': os.path.join(checkpoint_dir, 's0', 'best.ckpt'),
            's1': os.path.join(checkpoint_dir, 's1', 'best.ckpt'),
            's2': os.path.join(checkpoint_dir, 's2', 'best.ckpt'),
        }
    else:
        checkpoint_paths = {
            's0': os.path.join(checkpoint_dir, 's0', 'best.ckpt'),
            's1': os.path.join(checkpoint_dir, 's1', 'best.ckpt'),
            's2': os.path.join(checkpoint_dir, 's2', 'best.ckpt'),
        }

    # Verify checkpoints exist
    for run_name, ckpt_path in checkpoint_paths.items():
        if not os.path.isfile(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

    # Transform
    transform = get_test_transform()

    # ID dataset: CIFAR-10 test
    id_imglist = os.path.join(imglist_dir, 'test_cifar10.txt')
    if not os.path.isfile(id_imglist):
        print(f"Error: ID imglist not found: {id_imglist}", file=sys.stderr)
        sys.exit(1)

    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=None,
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)

    # OOD datasets: CIFAR-100 and TinyImageNet
    ood_configs = [
        ('cifar100', os.path.join(ood_imglist_dir, 'cifar100', 'test_cifar100.txt')),
        ('tin', os.path.join(ood_imglist_dir, 'tin', 'test_tin.txt')),
    ]

    ood_datasets = {}
    ood_loaders = {}
    for ood_name, ood_imglist_path in ood_configs:
        if not os.path.isfile(ood_imglist_path):
            print(f"Error: OOD imglist not found: {ood_imglist_path}", file=sys.stderr)
            sys.exit(1)
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=ood_imglist_path,
            data_dir=data_dir,
            num_classes=10,  # Not used for OOD, but required by ImglistDataset
            preprocessor=transform,
            data_aux_preprocessor=None,
        )
        ood_datasets[ood_name] = ood_dataset
        ood_loaders[ood_name] = DataLoader(ood_dataset, batch_size=200, shuffle=False, num_workers=0)

    # Evaluate each run
    run_metrics = {}
    for run_name in ['s0', 's1', 's2']:
        model = load_model(checkpoint_paths[run_name], device)

        # ID scores
        id_scores, _ = compute_ebo_scores(model, id_loader, temperature=1.0, device=device)

        # OOD scores for each dataset
        ood_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores, _ = compute_ebo_scores(model, ood_loader, temperature=1.0, device=device)
            auroc = compute_auroc(id_scores, ood_scores)
            ood_results[ood_name] = round(auroc, 2)

        run_metrics[run_name] = ood_results

    # Compute dataset mean within each run, then mean of runs
    dataset_names = ['cifar100', 'tin']
    dataset_means = {d: [] for d in dataset_names}
    for run_name in ['s0', 's1', 's2']:
        for d in dataset_names:
            dataset_means[d].append(run_metrics[run_name][d])

    # Dataset mean across runs
    dataset_mean_values = {d: np.mean(dataset_means[d]) for d in dataset_names}

    # Mean of dataset means (aggregation: dataset_mean_then_run_mean)
    actual = np.mean(list(dataset_mean_values.values()))

    # Sample counts
    dataset_counts = {
        'cifar100': len(ood_datasets['cifar100']),
        'tin': len(ood_datasets['tin']),
    }

    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(float(actual), 2),
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    # Print exactly one strict-JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
