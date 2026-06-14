#!/usr/bin/env python3
"""Reproduce EBO Near-OOD AUROC for CIFAR-10 using OpenOOD checkpoints."""

import argparse
import json
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# Constants from openood/preprocessors/transform.py
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ---------------------------------------------------------------------------
# Test transform: Resize(32) -> CenterCrop(32) -> ToTensor -> Normalize
# This matches TestStandardPreProcessor for CIFAR-10 (image_size=32, pre_size=32)
# ---------------------------------------------------------------------------
test_transform = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Dataset info from openood/evaluation_api/datasets.py
# ---------------------------------------------------------------------------
DATA_INFO = {
    'cifar10': {
        'id': {
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'ood': {
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            }
        }
    }
}

NUM_CLASSES = 10


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points (0-100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    # Sort by score descending
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
    return auroc * 100.0


def compute_ebo_scores(net, loader, temperature=1.0):
    """Compute negative energy scores for all samples in loader."""
    net.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            data = batch['data']
            logits = net(data)
            # EBO: -E(x) = logsumexp(f(x)/T)
            energy = temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to results/cifar10_resnet18_32x32_base_e100_lr0.1_default')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    # -----------------------------------------------------------------------
    # Build ID and OOD datasets
    # -----------------------------------------------------------------------
    id_info = DATA_INFO['cifar10']['id']['test']
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(data_root, id_info['imglist_path']),
        data_dir=os.path.join(data_root, id_info['data_dir']),
        num_classes=NUM_CLASSES,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ood_loaders = {}
    ood_datasets_info = DATA_INFO['cifar10']['ood']['near']
    for ds_name in ood_datasets_info['datasets']:
        ds_info = ood_datasets_info[ds_name]
        ds = ImglistDataset(
            name=ds_name,
            imglist_pth=os.path.join(data_root, ds_info['imglist_path']),
            data_dir=os.path.join(data_root, ds_info['data_dir']),
            num_classes=NUM_CLASSES,
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform,
        )
        ood_loaders[ds_name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Iterate over seeds
    # -----------------------------------------------------------------------
    run_metrics = {}
    dataset_counts = {}

    for subfolder in sorted(glob.glob(os.path.join(root, 's*'))):
        seed_name = os.path.basename(subfolder)
        ckpt_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            continue

        # Load model
        net = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location='cpu')
        net.load_state_dict(state)
        net.eval()

        # ID scores
        id_scores = compute_ebo_scores(net, id_loader)

        # OOD scores per dataset
        seed_metrics = {}
        for ds_name, loader in ood_loaders.items():
            ood_scores = compute_ebo_scores(net, loader)
            auroc = compute_auroc(id_scores, ood_scores)
            seed_metrics[ds_name] = round(auroc, 2)
            dataset_counts[ds_name] = len(ood_scores)

        run_metrics[seed_name] = seed_metrics

    # -----------------------------------------------------------------------
    # Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    dataset_names = list(ood_loaders.keys())
    run_dataset_means = []
    for seed_name, metrics in run_metrics.items():
        ds_values = [metrics[ds] for ds in dataset_names]
        run_dataset_means.append(np.mean(ds_values))
    actual = float(np.mean(run_dataset_means))

    # Build output
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 2),
        'datasets': {ds: dataset_counts[ds] for ds in dataset_names},
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print('REPRO_RESULT ' + json.dumps(result))


if __name__ == '__main__':
    main()
