#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD checkpoints."""

import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Import only the required modules from openood
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors.transform import normalization_dict, interpolation_modes, Convert


def get_test_transform():
    """Build the test transform exactly as TestStandardPreProcessor does."""
    pre_size = 32
    image_size = 32
    interpolation = interpolation_modes['bilinear']
    mean, std = normalization_dict['cifar10']
    transform = tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(pre_size, interpolation=interpolation),
        tvs_trans.CenterCrop(image_size),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])
    return transform


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points.
    
    Positive class: OOD (higher score = more OOD-like).
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    
    # Compute TPR and FPR
    pos_count = np.sum(sorted_labels == 1)
    neg_count = np.sum(sorted_labels == 0)
    
    if pos_count == 0 or neg_count == 0:
        return 50.0
    
    tpr = np.cumsum(sorted_labels == 1) / pos_count
    fpr = np.cumsum(sorted_labels == 0) / neg_count
    
    # Add (0,0) point
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
    
    # Compute AUC using trapezoidal rule
    auroc = np.trapz(tpr, fpr) * 100.0
    return auroc


def energy_score(logits, temperature=1.0):
    """Compute energy score (negative free energy)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace',
                        help='Root directory containing data and checkpoints')
    args = parser.parse_args()
    
    root = args.root
    data_dir = os.path.join(root, 'data', 'images_classic')
    checkpoint_dir = root
    
    # Checkpoint paths
    checkpoint_paths = {
        's0': os.path.join(checkpoint_dir, 's0', 'best.ckpt'),
        's1': os.path.join(checkpoint_dir, 's1', 'best.ckpt'),
        's2': os.path.join(checkpoint_dir, 's2', 'best.ckpt'),
    }
    
    # Verify checkpoints exist
    for seed, path in checkpoint_paths.items():
        if not os.path.exists(path):
            print(f"ERROR: Checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
    
    # Dataset paths
    id_imglist = os.path.join(root, 'data', 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    ood_datasets = {
        'cifar100': os.path.join(root, 'data', 'benchmark_imglist', 'cifar100', 'test_cifar100.txt'),
        'tin': os.path.join(root, 'data', 'benchmark_imglist', 'tinyimagenet', 'test_tinyimagenet.txt'),
    }
    # If the above paths don't exist, try relative to root/data
    if not os.path.exists(id_imglist):
        id_imglist = os.path.join(root, 'data', 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
        ood_datasets = {
            'cifar100': os.path.join(root, 'data', 'benchmark_imglist', 'cifar100', 'test_cifar100.txt'),
            'tin': os.path.join(root, 'data', 'benchmark_imglist', 'tinyimagenet', 'test_tinyimagenet.txt'),
        }
    
    # Verify dataset files exist
    for name, path in [('ID', id_imglist)] + list(ood_datasets.items()):
        if not os.path.exists(path):
            print(f"ERROR: Dataset list not found: {path}", file=sys.stderr)
            sys.exit(1)
    
    # Build transform
    transform = get_test_transform()
    
    # Create ID dataset
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    
    # Create OOD datasets
    ood_datasets_loaded = {}
    for ood_name, imglist_pth in ood_datasets.items():
        ood_datasets_loaded[ood_name] = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=10,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
    
    # DataLoader parameters
    loader_params = {
        'batch_size': 200,
        'shuffle': False,
        'num_workers': 0,  # CPU-only
        'pin_memory': False,
    }
    
    id_loader = DataLoader(id_dataset, **loader_params)
    ood_loaders = {name: DataLoader(ds, **loader_params) for name, ds in ood_datasets_loaded.items()}
    
    # Store results per run
    run_metrics = {}
    dataset_counts = {}
    
    for seed_name in ['s0', 's1', 's2']:
        # Load model
        model = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(checkpoint_paths[seed_name], map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        elif 'net' in state_dict:
            model.load_state_dict(state_dict['net'])
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        
        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch['data']
                logits = model(images)
                scores = energy_score(logits)
                id_scores.append(scores.cpu().numpy())
        
        id_scores = np.concatenate(id_scores)
        
        # Compute OOD scores for each dataset
        ood_results = {}
        for ood_name, ood_loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch['data']
                    logits = model(images)
                    scores = energy_score(logits)
                    ood_scores.append(scores.cpu().numpy())
            
            ood_scores = np.concatenate(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            ood_results[ood_name] = auroc
            dataset_counts[ood_name] = len(ood_scores)
        
        run_metrics[seed_name] = ood_results
    
    # Compute dataset mean within each run, then mean of runs
    dataset_means = {}
    for ood_name in ood_datasets:
        values = [run_metrics[seed][ood_name] for seed in ['s0', 's1', 's2']]
        dataset_means[ood_name] = np.mean(values)
    
    # Mean of dataset means
    actual = np.mean(list(dataset_means.values()))
    
    # Build result
    result = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {name: int(dataset_counts[name]) for name in ood_datasets},
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    
    # Print exactly one JSON line
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
