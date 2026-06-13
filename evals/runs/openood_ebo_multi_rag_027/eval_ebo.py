#!/usr/bin/env python3
"""Reproduce EBO Near-OOD AUROC for CIFAR-10 with ResNet18_32x32.

Executes evaluation for seeds s0, s1, s2 using official checkpoints,
computes EBO scores and AUROC locally, and prints a single JSON result.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms

# Direct imports from openood modules (no evaluation_api, evaluators, or postprocessors)
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset


def get_test_transform():
    """Construct the test transform for CIFAR-10 from OpenOOD's base_preprocessor.

    This replicates the transform used in openood/preprocessors/transform.py
    for the 'base_preprocessor' with normalization_type='cifar10'.
    The correct normalization statistics for CIFAR-10 are:
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.247, 0.2435, 0.2616]
    """
    return transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.2435, 0.2616]),
    ])


def compute_auroc(conf_scores, labels):
    """Compute AUROC (percentage) following OpenOOD's convention.

    OOD samples have label == -1, ID samples have label >= 0.
    The postprocessor assumes ID samples have larger confidence values.
    """
    ood_indicator = np.zeros_like(labels)
    ood_indicator[labels == -1] = 1
    # Negate confidence because we treat OOD as positive and ID conf > OOD conf
    fpr, tpr, _ = metrics.roc_curve(ood_indicator, -conf_scores)
    auroc = metrics.auc(fpr, tpr)
    return auroc * 100.0  # Convert to percentage


def evaluate_seed(seed_dir, id_data_name, ood_datasets, batch_size, data_root, imglist_root):
    """Evaluate a single seed checkpoint on ID and OOD datasets.

    Returns dict mapping OOD dataset name -> AUROC percentage.
    """
    checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    # Load model
    net = ResNet18_32x32(num_classes=10)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Handle possible 'net.' prefix in state dict keys
    if any(k.startswith('net.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[4:] if k.startswith('net.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    net.load_state_dict(state_dict)
    net.eval()

    transform = get_test_transform()

    # Build ID dataset (CIFAR-10 test split)
    id_imglist = os.path.join(imglist_root, id_data_name, f'test_{id_data_name}.txt')
    # The ID dataset uses num_classes=10; labels are 0-9, no soft_label issue.
    id_dataset = ImglistDataset(
        name='id',
        imglist_pth=id_imglist,
        data_dir=data_root,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    for ood_name in ood_datasets:
        ood_imglist = os.path.join(imglist_root, ood_name, f'test_{ood_name}.txt')
        # Determine num_classes for OOD dataset based on its name
        if ood_name == 'cifar100':
            ood_num_classes = 100
        elif ood_name == 'tin':
            ood_num_classes = 200
        else:
            ood_num_classes = 10
        # The OOD image list file is located under the ID dataset's imglist directory
        ood_imglist = os.path.join(imglist_root, id_data_name, f'test_{ood_name}.txt')
        ood_dataset = ImglistDataset(
            name='ood',
            imglist_pth=ood_imglist,
            data_dir=data_root,
            num_classes=ood_num_classes,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Collect ID predictions and EBO scores
        id_conf_list = []
        id_pred_list = []
        id_label_list = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']
                label = batch['label']
                output = net(data)
                # EBO score: temperature * logsumexp(logits / temperature, dim=1)
                temperature = 1.0
                conf = temperature * torch.logsumexp(output / temperature, dim=1)
                score = torch.softmax(output, dim=1)
                _, pred = torch.max(score, dim=1)
                id_conf_list.append(conf.cpu().numpy())
                id_pred_list.append(pred.cpu().numpy())
                id_label_list.append(label.cpu().numpy())

        id_conf = np.concatenate(id_conf_list)
        id_pred = np.concatenate(id_pred_list)
        id_label = np.concatenate(id_label_list)

        # Collect OOD predictions and EBO scores
        ood_conf_list = []
        ood_pred_list = []
        ood_label_list = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data']
                label = batch['label']
                output = net(data)
                temperature = 1.0
                conf = temperature * torch.logsumexp(output / temperature, dim=1)
                score = torch.softmax(output, dim=1)
                _, pred = torch.max(score, dim=1)
                ood_conf_list.append(conf.cpu().numpy())
                ood_pred_list.append(pred.cpu().numpy())
                ood_label_list.append(label.cpu().numpy())

        ood_conf = np.concatenate(ood_conf_list)
        ood_pred = np.concatenate(ood_pred_list)
        ood_label = np.concatenate(ood_label_list)

        # Combine ID and OOD for AUROC computation
        all_conf = np.concatenate([id_conf, ood_conf])
        all_label = np.concatenate([id_label, ood_label])

        auroc = compute_auroc(all_conf, all_label)
        results[ood_name] = auroc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    parser.add_argument('--data-root', type=str, default='./data/images_classic',
                        help='Root directory for image data')
    parser.add_argument('--imglist-root', type=str, default='./data/benchmark_imglist',
                        help='Root directory for benchmark image lists')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    root = args.root
    data_root = args.data_root
    imglist_root = args.imglist_root
    batch_size = args.batch_size

    # Validate root structure
    seed_dirs = sorted([d for d in os.listdir(root) if d.startswith('s') and os.path.isdir(os.path.join(root, d))])
    if len(seed_dirs) == 0:
        print(f'ERROR: No seed subdirectories (s0, s1, s2) found in {root}', file=sys.stderr)
        sys.exit(1)

    # Near-OOD datasets for CIFAR-10
    ood_datasets = ['cifar100', 'tin']

    # Evaluate each seed
    run_metrics = {}
    for seed_name in seed_dirs:
        seed_dir = os.path.join(root, seed_name)
        try:
            results = evaluate_seed(seed_dir, 'cifar10', ood_datasets, batch_size, data_root, imglist_root)
            run_metrics[seed_name] = results
        except Exception as e:
            print(f'ERROR evaluating {seed_name}: {e}', file=sys.stderr)
            sys.exit(1)

    # Compute dataset-then-run mean
    # First, compute mean AUROC per dataset across runs
    dataset_means = {}
    dataset_counts = {}
    for ood_name in ood_datasets:
        values = [run_metrics[seed][ood_name] for seed in seed_dirs if ood_name in run_metrics[seed]]
        if values:
            dataset_means[ood_name] = np.mean(values)
            dataset_counts[ood_name] = len(values)
        else:
            dataset_means[ood_name] = 0.0
            dataset_counts[ood_name] = 0

    # Then, compute mean of dataset means
    actual = np.mean(list(dataset_means.values()))

    # Build output JSON
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {ood: dataset_counts[ood] for ood in ood_datasets},
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
