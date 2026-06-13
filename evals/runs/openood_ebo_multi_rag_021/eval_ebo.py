#!/usr/bin/env python3
"""
eval_ebo.py - CPU-safe EBO Near-OOD AUROC evaluation for CIFAR-10.

Reproduces official OpenOOD EBO results using ResNet18_32x32 checkpoints
(s0, s1, s2) on CIFAR-100 and TinyImageNet near-OOD datasets.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader

from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32


def get_preprocessor():
    """Return CIFAR-10 standard normalization transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])


def compute_auroc(id_conf, ood_conf):
    """Compute AUROC treating OOD as positive class.
    
    Following OpenOOD convention: ID samples have higher conf values,
    so we negate conf for ROC curve.
    """
    labels = np.concatenate([
        np.zeros(len(id_conf)),   # ID = 0
        np.ones(len(ood_conf))    # OOD = 1
    ])
    scores = np.concatenate([id_conf, ood_conf])
    fpr, tpr, _ = metrics.roc_curve(labels, -scores)  # negate for higher=ID
    return metrics.auc(fpr, tpr) * 100  # percentage points


def eval_run(net, id_loader, ood_loaders, device='cpu'):
    """Evaluate a single run (seed) and return per-dataset AUROCs."""
    net.eval()
    
    # Collect ID predictions and EBO scores
    id_scores = []
    id_labels = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            labels = batch['label'].numpy()
            logits = net(data)
            # EBO score: temperature * logsumexp(logits / temperature)
            # temperature = 1 (default)
            scores = torch.logsumexp(logits, dim=1).cpu().numpy()
            id_scores.append(scores)
            id_labels.append(labels)
    
    id_scores = np.concatenate(id_scores)
    id_labels = np.concatenate(id_labels)
    
    results = {}
    for dataset_name, ood_loader in ood_loaders.items():
        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                logits = net(data)
                scores = torch.logsumexp(logits, dim=1).cpu().numpy()
                ood_scores.append(scores)
        
        ood_scores = np.concatenate(ood_scores)
        auroc = compute_auroc(id_scores, ood_scores)
        results[dataset_name] = auroc
    
    return results


def main():
    parser = argparse.ArgumentParser(description='EBO Near-OOD AUROC evaluation')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0, s1, s2 subfolders')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size for DataLoader')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader workers (0 for CPU safety)')
    args = parser.parse_args()
    
    root = args.root
    device = 'cpu'
    
    # Verify root structure
    seed_dirs = sorted(glob.glob(os.path.join(root, 's*')))
    if not seed_dirs:
        print(f'ERROR: No seed directories (s0, s1, s2) found in {root}')
        sys.exit(1)
    
    # Setup data paths
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    imglist_root = os.path.join(data_root, 'benchmark_imglist', 'cifar10')
    images_root = os.path.join(data_root, 'images_classic')
    
    # ID dataset (CIFAR-10 test)
    preprocessor = get_preprocessor()
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(imglist_root, 'test_cifar10.txt'),
        data_dir=images_root,
        num_classes=10,
        preprocessor=preprocessor,
        data_aux_preprocessor=preprocessor
    )
    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Near-OOD datasets
    ood_configs = {
        'cifar100': {
            'imglist': os.path.join(imglist_root, 'test_cifar100.txt'),
            'data_dir': images_root
        },
        'tin': {
            'imglist': os.path.join(imglist_root, 'test_tin.txt'),
            'data_dir': images_root
        }
    }
    
    ood_loaders = {}
    for name, cfg in ood_configs.items():
        dataset = ImglistDataset(
            name=f'{name}_test',
            imglist_pth=cfg['imglist'],
            data_dir=images_root,
            num_classes=10,  # not used for OOD
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor
        )
        ood_loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Evaluate each seed
    run_metrics = {}
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
        
        if not os.path.exists(checkpoint_path):
            print(f'WARNING: Checkpoint not found at {checkpoint_path}, skipping')
            continue
        
        # Load model
        net = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        
        print(f'Evaluating {seed_name}...')
        results = eval_run(net, id_loader, ood_loaders, device)
        run_metrics[seed_name] = results
        
        print(f'  {seed_name} results:')
        for ds_name, auroc in results.items():
            print(f'    {ds_name}: {auroc:.2f}')
    
    # Aggregate results
    if not run_metrics:
        print('ERROR: No runs completed successfully')
        sys.exit(1)
    
    # Compute dataset mean within each run, then mean of runs
    dataset_aurocs = {ds: [] for ds in ood_configs}
    for seed_name, results in run_metrics.items():
        for ds_name, auroc in results.items():
            dataset_aurocs[ds_name].append(auroc)
    
    # Dataset mean across runs
    dataset_means = {}
    for ds_name, values in dataset_aurocs.items():
        dataset_means[ds_name] = np.mean(values)
    
    # Run means (mean of dataset means per run)
    run_means = []
    for seed_name, results in run_metrics.items():
        run_mean = np.mean(list(results.values()))
        run_means.append(run_mean)
    
    actual = np.mean(run_means)
    
    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': round(float(actual), 2),
        'datasets': {
            'cifar100': len(run_metrics),
            'tin': len(run_metrics)
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }
    
    # Print required output line
    print(f'REPRO_RESULT {output}')


if __name__ == '__main__':
    main()
