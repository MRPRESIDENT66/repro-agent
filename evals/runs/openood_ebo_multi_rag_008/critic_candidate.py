#!/usr/bin/env python3
"""CPU-safe EBO evaluation for CIFAR-10 ID / CIFAR-100 & TinyImageNet OOD."""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import OpenOOD's exact ResNet18_32x32 implementation
from openood.networks import ResNet18_32x32

# ---------------------------------------------------------------------------
# CIFAR-10 test preprocessing (from openood/evaluation_api/preprocessor.py)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Dataset loader from benchmark imglist
# ---------------------------------------------------------------------------
def load_imglist(imglist_path, data_dir):
    """Return list of (full_path, label) pairs."""
    samples = []
    with open(imglist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            img_name = parts[0]
            extra = parts[1] if len(parts) > 1 else '0'
            full_path = os.path.join(data_dir, img_name)
            # label is the integer after the image name
            try:
                label = int(extra)
            except ValueError:
                label = 0
            samples.append((full_path, label))
    return samples

def build_loader(imglist_path, data_dir, batch_size=200, shuffle=False):
    """Return DataLoader for a benchmark imglist."""
    samples = load_imglist(imglist_path, data_dir)
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)
        def __getitem__(self, idx):
            path, label = samples[idx]
            img = Image.open(path).convert('RGB')
            x = test_transform(img)
            return x, label
    return torch.utils.data.DataLoader(
        _Dataset(), batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=False)

# ---------------------------------------------------------------------------
# EBO score computation (logsumexp confidence)
# ---------------------------------------------------------------------------
def compute_ebo_scores(net, loader, temperature=1.0, device='cpu'):
    """Return numpy array of EBO energy scores for all samples."""
    net.eval()
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = net(x)
            # energy = temperature * logsumexp(logits / temperature)
            energy = temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores)

# ---------------------------------------------------------------------------
# AUROC computation (from openood/evaluators/metrics.py semantics)
# Higher energy => more OOD-like
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Return AUROC where higher energy => more OOD-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    # sort descending (higher energy = more OOD)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 0.5
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # trapezoidal integration
    auroc = np.trapz(tpr, fpr)
    return auroc

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root dir containing s0/s1/s2 subdirs with best.ckpt')
    parser.add_argument('--data-dir', type=str, default='./data/images_classic',
                        help='Path to images_classic directory')
    parser.add_argument('--imglist-dir', type=str, default='./data/benchmark_imglist/cifar10',
                        help='Path to benchmark_imglist/cifar10 directory')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = 'cpu'
    num_classes = 10
    batch_size = 200

    # Paths
    id_imglist = os.path.join(args.imglist_dir, 'test_cifar10.txt')
    ood_imglists = {
        'cifar100': os.path.join(args.imglist_dir, 'test_cifar100.txt'),
        'tin': os.path.join(args.imglist_dir, 'test_tin.txt'),
    }

    # Build ID loader once
    id_loader = build_loader(id_imglist, args.data_dir, batch_size)

    # Build OOD loaders
    ood_loaders = {}
    for name, path in ood_imglists.items():
        ood_loaders[name] = build_loader(path, args.data_dir, batch_size)

    # Seeds
    seeds = ['s0', 's1', 's2']
    run_metrics = {}

    for seed in seeds:
        ckpt_path = os.path.join(args.root, seed, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        # Load model using OpenOOD's exact ResNet18_32x32 implementation
        net = ResNet18_32x32(num_classes=num_classes)
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        # ID scores
        id_scores = compute_ebo_scores(net, id_loader, args.temperature, device)

        # OOD scores per dataset
        auroc_dict = {}
        for name, loader in ood_loaders.items():
            ood_scores = compute_ebo_scores(net, loader, args.temperature, device)
            auroc = compute_auroc(id_scores, ood_scores)
            auroc_dict[name] = auroc

        run_metrics[seed] = auroc_dict

    # Aggregation: dataset mean within each run, then mean of runs
    # For each run, compute mean over datasets
    run_means = []
    for seed in seeds:
        vals = list(run_metrics[seed].values())
        run_mean = float(np.mean(vals))
        run_means.append(run_mean)

    actual = float(np.mean(run_means))

    # Build counts (number of samples per dataset)
    id_count = len(load_imglist(id_imglist, args.data_dir))
    ood_counts = {}
    for name in ood_imglists:
        ood_counts[name] = len(load_imglist(ood_imglists[name], args.data_dir))

    # Build result dict
    result = {
        "metric": "near_ood_auroc",
        "actual": actual,
        "datasets": {
            "cifar100": ood_counts['cifar100'],
            "tin": ood_counts['tin'],
        },
        "run_metrics": {
            seed: {
                "cifar100": run_metrics[seed]['cifar100'],
                "tin": run_metrics[seed]['tin'],
            }
            for seed in seeds
        },
        "aggregation": "dataset_mean_then_run_mean",
    }

    # Print exactly one evidence line
    print(f'REPRO_RESULT {json.dumps(result)}')

if __name__ == '__main__':
    main()
