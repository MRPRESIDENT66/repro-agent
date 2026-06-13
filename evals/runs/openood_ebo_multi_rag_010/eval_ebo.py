#!/usr/bin/env python3
"""EBO evaluation for CIFAR-10 near-OOD detection (CIFAR-100, TinyImageNet)."""

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

# Import OpenOOD's exact ResNet18_32x32 implementation
from openood.networks import ResNet18_32x32


# ---------------------------------------------------------------------------
# Dataset that reads image list files (like ImglistDataset)
# ---------------------------------------------------------------------------
class ImglistDataset(Dataset):
    def __init__(self, imglist_pth, data_dir, transform):
        super().__init__()
        with open(imglist_pth) as f:
            self.imglist = f.readlines()
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        line = self.imglist[idx].strip()
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        path = os.path.join(self.data_dir, image_name)
        image = Image.open(path).convert('RGB')
        x = self.transform(image)
        # label is the integer after the image name
        label = int(extra_str)
        return x, label


# ---------------------------------------------------------------------------
# CIFAR-10 test preprocessing (from openood/evaluation_api/preprocessor.py)
# ---------------------------------------------------------------------------
def get_cifar10_test_transform():
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616]),
    ])


# ---------------------------------------------------------------------------
# AUROC computation (from openood/evaluators/metrics.py)
# ---------------------------------------------------------------------------
def compute_auroc(conf_in, conf_out):
    """Compute AUROC in percentage points."""
    scores = np.concatenate([conf_in, conf_out])
    labels = np.concatenate([np.ones(len(conf_in)), np.zeros(len(conf_out))])
    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # Trapezoidal integration
    auroc = np.trapz(tpr, fpr)
    return auroc * 100.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='Checkpoint directory containing s0/s1/s2/best.ckpt')
    args = parser.parse_args()

    root = args.root
    device = torch.device('cpu')

    # Preprocessing
    transform = get_cifar10_test_transform()

    # Data directories - look relative to workspace root
    # Try multiple possible locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)  # parent of script dir
    
    # Possible data locations
    possible_data_dirs = [
        os.path.join(workspace_dir, 'data'),
        os.path.join(workspace_dir, '..', 'data'),
        '/workspace/data',
        os.path.join(os.path.expanduser('~'), 'data'),
    ]
    
    data_dir = None
    imglist_dir = None
    
    for base in possible_data_dirs:
        candidate_data = os.path.join(base, 'images_classic')
        candidate_imglist = os.path.join(base, 'benchmark_imglist', 'cifar10')
        if os.path.isdir(candidate_data) and os.path.isdir(candidate_imglist):
            data_dir = candidate_data
            imglist_dir = candidate_imglist
            break
    
    if data_dir is None:
        # Fallback: try to find data relative to root
        for base in [root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))]:
            candidate_data = os.path.join(base, 'data', 'images_classic')
            candidate_imglist = os.path.join(base, 'data', 'benchmark_imglist', 'cifar10')
            if os.path.isdir(candidate_data) and os.path.isdir(candidate_imglist):
                data_dir = candidate_data
                imglist_dir = candidate_imglist
                break
    
    if data_dir is None:
        print('Error: Could not find data directory. Please ensure data is in ./data/ relative to workspace.', file=sys.stderr)
        sys.exit(1)

    # Datasets
    id_dataset = ImglistDataset(
        os.path.join(imglist_dir, 'test_cifar10.txt'),
        data_dir, transform)
    ood_datasets = {
        'cifar100': ImglistDataset(
            os.path.join(imglist_dir, 'test_cifar100.txt'),
            data_dir, transform),
        'tin': ImglistDataset(
            os.path.join(imglist_dir, 'test_tin.txt'),
            data_dir, transform),
    }

    # DataLoaders
    batch_size = 200
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=False)
    ood_loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=2, pin_memory=False)
                   for name, ds in ood_datasets.items()}

    # Runs
    run_names = ['s0', 's1', 's2']
    run_metrics = {}

    for run in run_names:
        ckpt_path = os.path.join(root, run, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f'Checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)

        # Load model using OpenOOD's exact ResNet18_32x32
        model = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location=device)
        # Handle possible 'net.' prefix or direct state_dict
        if 'state_dict' in state:
            state = state['state_dict']
        # Remove 'module.' prefix if present
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)
        model.to(device)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for x, _ in id_loader:
                x = x.to(device)
                logits = model(x)
                # EBO energy score: temperature * logsumexp(logits / temperature)
                temperature = 1.0
                energy = temperature * torch.logsumexp(logits / temperature, dim=1)
                id_scores.append(energy.cpu().numpy())
        id_scores = np.concatenate(id_scores)

        # Compute OOD scores per dataset
        ood_aurocs = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for x, _ in ood_loader:
                    x = x.to(device)
                    logits = model(x)
                    temperature = 1.0
                    energy = temperature * torch.logsumexp(logits / temperature, dim=1)
                    ood_scores.append(energy.cpu().numpy())
            ood_scores = np.concatenate(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            ood_aurocs[ood_name] = auroc

        run_metrics[run] = ood_aurocs

    # Aggregate: dataset mean within each run, then mean of runs
    # First compute per-run dataset mean
    run_dataset_means = []
    for run in run_names:
        vals = list(run_metrics[run].values())
        run_dataset_means.append(np.mean(vals))
    actual = np.mean(run_dataset_means)

    # Counts
    counts = {}
    for name, ds in ood_datasets.items():
        counts[name] = len(ds)

    # Build output
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {'cifar100': counts['cifar100'], 'tin': counts['tin']},
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
