#!/usr/bin/env python3
"""EBO OOD evaluation for CIFAR-10 ResNet18_32x32 checkpoints (CPU-safe)."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 1. Minimal ResNet18_32x32 (from openood/networks/resnet18_32x32.py)
# ---------------------------------------------------------------------------
class ResNet18_32x32(torch.nn.Module):
    """ResNet-18 for 32x32 images (CIFAR-10 sized)."""
    def __init__(self, num_classes=10):
        super().__init__()
        from torchvision.models.resnet import BasicBlock, ResNet
        self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        # Adjust first conv for 32x32
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = torch.nn.Identity()

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# 2. Dataset for image list files (same as OpenOOD ImglistDataset)
# ---------------------------------------------------------------------------
class ImglistDataset(Dataset):
    """Load images from a benchmark_imglist file."""
    def __init__(self, imglist_pth, data_dir, transform):
        with open(imglist_pth) as f:
            self.imglist = [line.strip() for line in f.readlines()]
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        line = self.imglist[idx]
        tokens = line.split(' ', 1)
        img_name = tokens[0]
        path = os.path.join(self.data_dir, img_name)
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# ---------------------------------------------------------------------------
# 3. CIFAR-10 test preprocessing (from evaluation_api/preprocessor.py)
# ---------------------------------------------------------------------------
def get_cifar10_transform():
    return transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

# ---------------------------------------------------------------------------
# 4. EBO postprocessor (from openood/postprocessors/ebo_postprocessor.py)
# ---------------------------------------------------------------------------
def ebo_confidence(logits, temperature=1.0):
    """Compute EBO energy score."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# 5. AUROC computation (from openood/evaluators/metrics.py)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC given ID and OOD confidence scores."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    # True positive rate and false positive rate
    tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return auroc

# ---------------------------------------------------------------------------
# 6. Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_ebo(checkpoint_root, device='cpu'):
    """Run EBO evaluation for all seeds and OOD datasets."""
    transform = get_cifar10_transform()
    data_dir = './data/images_classic/'
    imglist_dir = './data/benchmark_imglist/'

    # ID dataset: CIFAR-10 test
    id_dataset = ImglistDataset(
        os.path.join(imglist_dir, 'cifar10', 'test_cifar10.txt'),
        data_dir, transform)

    # OOD datasets: CIFAR-100 and TinyImageNet
    ood_datasets = {
        'cifar100': ImglistDataset(
            os.path.join(imglist_dir, 'cifar100', 'test_cifar100.txt'),
            data_dir, transform),
        'tin': ImglistDataset(
            os.path.join(imglist_dir, 'tin', 'test_tin.txt'),
            data_dir, transform),
    }

    # Find all seed directories (s0, s1, s2, ...)
    seed_dirs = sorted([d for d in os.listdir(checkpoint_root)
                        if d.startswith('s') and os.path.isdir(os.path.join(checkpoint_root, d))])
    if not seed_dirs:
        print(f"ERROR: No seed directories found in {checkpoint_root}", file=sys.stderr)
        sys.exit(1)

    # Store per-seed, per-dataset AUROC
    all_aurocs = []  # list of dicts: {dataset: auroc}

    for seed_dir in seed_dirs:
        ckpt_path = os.path.join(checkpoint_root, seed_dir, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found: {ckpt_path}", file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location=device)
        # Handle possible 'net.' prefix
        if any(k.startswith('net.') for k in state_dict.keys()):
            model.load_state_dict(state_dict)
        else:
            model.net.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(id_dataset, batch_size=200, shuffle=False):
                batch = batch.to(device)
                logits = model(batch)
                scores = ebo_confidence(logits, temperature=1.0)
                id_scores.extend(scores.cpu().numpy())
        id_scores = np.array(id_scores)

        # Compute OOD scores per dataset
        seed_aurocs = {}
        for ood_name, ood_dataset in ood_datasets.items():
            ood_scores = []
            with torch.no_grad():
                for batch in torch.utils.data.DataLoader(ood_dataset, batch_size=200, shuffle=False):
                    batch = batch.to(device)
                    logits = model(batch)
                    scores = ebo_confidence(logits, temperature=1.0)
                    ood_scores.extend(scores.cpu().numpy())
            ood_scores = np.array(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            seed_aurocs[ood_name] = auroc

        all_aurocs.append(seed_aurocs)
        print(f"  Seed {seed_dir}: cifar100 AUROC={seed_aurocs['cifar100']:.4f}, tin AUROC={seed_aurocs['tin']:.4f}")

    if not all_aurocs:
        print("ERROR: No valid checkpoints found.", file=sys.stderr)
        sys.exit(1)

    # Aggregate: dataset mean within each run, then mean of runs
    # First, compute per-run dataset mean
    run_means = []
    for seed_aurocs in all_aurocs:
        run_mean = np.mean(list(seed_aurocs.values()))
        run_means.append(run_mean)
    actual = np.mean(run_means)

    # Compute per-dataset mean across runs
    dataset_aurocs = {}
    for ood_name in ood_datasets.keys():
        vals = [seed[ood_name] for seed in all_aurocs]
        dataset_aurocs[ood_name] = np.mean(vals)

    # Build run_metrics dict
    run_metrics = {}
    for i, seed_dir in enumerate(seed_dirs):
        run_metrics[seed_dir] = {
            'cifar100': all_aurocs[i]['cifar100'],
            'tin': all_aurocs[i]['tin'],
        }

    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': len(id_dataset),
            'tin': len(ood_datasets['tin']),
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    return result

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EBO OOD Evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing seed subdirectories (s0, s1, ...)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    result = evaluate_ebo(args.root, device=args.device)
    print(f'REPRO_RESULT {json.dumps(result)}')
