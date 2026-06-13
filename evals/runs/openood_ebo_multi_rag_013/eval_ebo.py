#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC for CIFAR-10 (ResNet18_32x32)

Reproduces official OpenOOD EBO results using s0/s1/s2 checkpoints.
Prints the required REPRO_RESULT line.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Minimal ResNet18_32x32 (exact OpenOOD architecture)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet18_32x32(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.mean(dim=(2, 3))  # global avg pool
        out = self.linear(out)
        return out


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_image_list(path):
    """Load image list file: each line is 'path label'."""
    with open(path, 'r') as f:
        lines = f.readlines()
    paths, labels = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            paths.append(parts[0])
            labels.append(int(parts[1]))
    return paths, labels


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, root, paths, transform=None):
        self.root = root
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.paths[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# EBO postprocessor
# ---------------------------------------------------------------------------
def ebo_score(logits, temperature=1.0):
    """Compute EBO confidence (higher = more ID)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation (percentage, ID=1, OOD=-1)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    from sklearn.metrics import roc_curve, auc
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)),
                             -np.ones(len(ood_scores))])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return 100.0 * auc(fpr, tpr)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/images_classic',
                        help='Root directory for image data')
    parser.add_argument('--checkpoint_root', type=str,
                        default='./results/cifar10_resnet18_32x32_base_e100_lr0.1_default',
                        help='Root directory containing s0/s1/s2 checkpoints')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='CPU-safe: use 0')
    args = parser.parse_args()

    device = torch.device('cpu')

    # -----------------------------------------------------------------------
    # Preprocessing (exact OpenOOD base_preprocessor)
    # -----------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    # -----------------------------------------------------------------------
    # ID data: CIFAR-10 test set
    # -----------------------------------------------------------------------
    id_root = os.path.join(args.root, 'cifar10')
    id_list_path = os.path.join(args.root, 'benchmark_imglist', 'cifar10',
                                'test_cifar10.txt')
    id_paths, _ = load_image_list(id_list_path)
    id_dataset = ImageListDataset(id_root, id_paths, transform=transform)
    id_loader = DataLoader(id_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)

    # -----------------------------------------------------------------------
    # OOD datasets: CIFAR-100 and TinyImageNet (near-OOD)
    # -----------------------------------------------------------------------
    ood_datasets = {
        'cifar100': {
            'root': os.path.join(args.root, 'cifar100'),
            'list': os.path.join(args.root, 'benchmark_imglist', 'cifar10',
                                 'test_cifar100.txt'),
        },
        'tin': {
            'root': os.path.join(args.root, 'tinyimagenet'),
            'list': os.path.join(args.root, 'benchmark_imglist', 'cifar10',
                                 'test_tin.txt'),
        },
    }

    ood_loaders = {}
    for name, info in ood_datasets.items():
        paths, _ = load_image_list(info['list'])
        dataset = ImageListDataset(info['root'], paths, transform=transform)
        ood_loaders[name] = DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)

    # -----------------------------------------------------------------------
    # Runs: s0, s1, s2
    # -----------------------------------------------------------------------
    run_names = ['s0', 's1', 's2']
    temperature = 1.0

    # Store per-run, per-dataset AUROC
    run_metrics = {run: {} for run in run_names}

    for run in run_names:
        # Load checkpoint
        ckpt_path = os.path.join(args.checkpoint_root, run, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)

        net = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location=device)
        # Handle possible 'state_dict' key
        if 'state_dict' in state:
            state = state['state_dict']
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                batch = batch.to(device)
                logits = net(batch)
                scores = ebo_score(logits, temperature)
                id_scores.extend(scores.cpu().numpy().tolist())
        id_scores = np.array(id_scores)

        # Compute OOD scores for each dataset
        for ood_name, loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = net(batch)
                    scores = ebo_score(logits, temperature)
                    ood_scores.extend(scores.cpu().numpy().tolist())
            ood_scores = np.array(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run][ood_name] = auroc

    # -----------------------------------------------------------------------
    # Aggregation: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    dataset_names = list(ood_datasets.keys())
    dataset_means = {}
    for dname in dataset_names:
        vals = [run_metrics[r][dname] for r in run_names]
        dataset_means[dname] = np.mean(vals)

    # Mean across datasets for each run
    run_means = []
    for r in run_names:
        vals = [run_metrics[r][dname] for dname in dataset_names]
        run_means.append(np.mean(vals))
    actual = np.mean(run_means)

    # -----------------------------------------------------------------------
    # Print required REPRO_RESULT line
    # -----------------------------------------------------------------------
    result = {
        "metric": "near_ood_auroc",
        "actual": round(float(actual), 4),
        "datasets": {dname: round(float(dataset_means[dname]), 4)
                     for dname in dataset_names},
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean"
    }
    # Convert run_metrics values to float for JSON
    for r in run_names:
        for d in dataset_names:
            result["run_metrics"][r][d] = round(float(run_metrics[r][d]), 4)

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
