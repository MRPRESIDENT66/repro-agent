#!/usr/bin/env python3
"""Reproduce EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32.

Usage:
    python eval_ebo.py --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.utils.config import Config


def compute_ebo_scores(logits, temperature=1.0):
    """Compute negative energy scores: -E(x) = logsumexp(f(x)/T)."""
    return torch.logsumexp(logits / temperature, dim=1)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC treating ID as positive class, return percentage."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    auroc = roc_auc_score(labels, scores)
    return auroc * 100.0


def load_model(checkpoint_path, num_classes=10):
    """Load ResNet18_32x32 from checkpoint."""
    model = ResNet18_32x32(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location='cpu')
    # Handle different checkpoint formats
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    elif 'net' in state:
        model.load_state_dict(state['net'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def get_preprocessor_from_config(config_path):
    """Create TestStandardPreProcessor from a config file.

    Uses the standard OpenOOD Config class to load the YAML file,
    then extracts preprocessor parameters from the resulting Config object.
    """
    from openood.utils.config import Config
    config = Config(config_path)
    # Build a simple object with attribute access for TestStandardPreProcessor
    class SimpleConfig:
        def __init__(self, d):
            self.__dict__.update(d)
    config_dict = {
        'pre_size': config.dataset.pre_size,
        'img_size': config.dataset.image_size,
        'normalization': config.dataset.normalization,
    }
    simple_config = SimpleConfig(config_dict)
    return TestStandardPreProcessor(simple_config)


def get_dataset(imglist_pth, data_dir, preprocessor, num_classes=10):
    """Create ImglistDataset for evaluation."""
    return ImglistDataset(
        name='eval',
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=num_classes,
        preprocessor=preprocessor,
        data_aux_preprocessor=preprocessor,
    )


def evaluate_model(model, dataloader, temperature=1.0):
    """Compute EBO scores for all samples in dataloader."""
    scores = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch['data']
            logits = model(data)
            energy = compute_ebo_scores(logits, temperature)
            scores.append(energy.cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    temperature = args.temperature

    # Data paths (from config.yml)
    data_dir = './data/images_classic/'
    id_imglist = './data/benchmark_imglist/cifar10/test_cifar10.txt'
    ood_imglists = {
        'cifar100': './data/benchmark_imglist/cifar100/test_cifar100.txt',
        'tin': './data/benchmark_imglist/tin/test_tin.txt',
    }

    # Get preprocessor from first seed's config
    s0_config = os.path.join(root, 's0', 'config.yml')
    preprocessor = get_preprocessor_from_config(s0_config)

    # Create ID dataset
    id_dataset = get_dataset(id_imglist, data_dir, preprocessor, num_classes=10)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create OOD datasets
    ood_datasets = {}
    ood_loaders = {}
    for name, imglist in ood_imglists.items():
        ds = get_dataset(imglist, data_dir, preprocessor, num_classes=10)
        ood_datasets[name] = ds
        ood_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Iterate over seeds
    run_metrics = {}
    all_dataset_aurocs = {name: [] for name in ood_datasets}

    for subfolder in sorted(glob.glob(os.path.join(root, 's*'))):
        seed_name = os.path.basename(subfolder)
        checkpoint_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: checkpoint not found at {checkpoint_path}, skipping")
            continue

        model = load_model(checkpoint_path, num_classes=10)

        # Compute ID scores
        id_scores = evaluate_model(model, id_loader, temperature)

        # Compute OOD scores and AUROC per dataset
        seed_metrics = {}
        for name, loader in ood_loaders.items():
            ood_scores = evaluate_model(model, loader, temperature)
            auroc = compute_auroc(id_scores, ood_scores)
            seed_metrics[name] = auroc
            all_dataset_aurocs[name].append(auroc)

        run_metrics[seed_name] = seed_metrics

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First compute per-run dataset mean
    run_dataset_means = []
    for seed_name, metrics in run_metrics.items():
        dataset_values = list(metrics.values())
        run_mean = np.mean(dataset_values)
        run_dataset_means.append(run_mean)

    # Then mean of runs
    actual = np.mean(run_dataset_means)

    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {name: len(ds) for name, ds in ood_datasets.items()},
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
