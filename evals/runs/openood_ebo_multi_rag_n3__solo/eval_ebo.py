#!/usr/bin/env python3
"""Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Direct imports from openood modules (no evaluation_api, evaluators, postprocessors)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.preprocessors.transform import Convert, interpolation_modes, normalization_dict


def get_test_transform():
    """Build the exact test transform from openood/preprocessors/transform.py
    and test_preprocessor.py: Convert('RGB'), Resize(pre_size), CenterCrop(image_size),
    ToTensor(), Normalize(mean, std)."""
    pre_size = 32
    image_size = 32
    interpolation = interpolation_modes['bilinear']
    mean = normalization_dict['cifar10'][0]
    std = normalization_dict['cifar10'][1]
    transform = tvs_trans.Compose([
        Convert('RGB'),
        tvs_trans.Resize(pre_size, interpolation=interpolation),
        tvs_trans.CenterCrop(image_size),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])
    return transform


def compute_ebo_scores(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute Energy-Based OOD score: -temperature * logsumexp(logits / temperature)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points. ID scores should be higher (lower energy) for ID."""
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    # EBO: lower score = more OOD-like. For AUROC we want higher score = more ID-like.
    # So we negate: higher -score = more ID-like.
    return roc_auc_score(y_true, -y_score) * 100.0


def evaluate_checkpoint(model: torch.nn.Module, id_loader: DataLoader,
                        ood_loaders: dict, device: torch.device) -> dict:
    """Evaluate a single checkpoint on ID and OOD datasets, return AUROC per OOD dataset."""
    model.eval()
    model.to(device)

    # Collect ID scores
    id_scores_list = []
    with torch.no_grad():
        for batch in id_loader:
            images = batch['data'].to(device)
            logits = model(images)
            scores = compute_ebo_scores(logits)
            id_scores_list.append(scores.cpu().numpy())
    id_scores = np.concatenate(id_scores_list)

    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_scores_list = []
        with torch.no_grad():
            for batch in ood_loader:
                images = batch['data'].to(device)
                logits = model(images)
                scores = compute_ebo_scores(logits)
                ood_scores_list.append(scores.cpu().numpy())
        ood_scores = np.concatenate(ood_scores_list)
        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace',
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    root = args.root
    data_dir = os.path.join(root, 'data', 'images_classic')
    imglist_dir = os.path.join(root, 'data', 'benchmark_imglist')
    checkpoint_dir = os.path.join(root, 'results',
                                  'cifar10_resnet18_32x32_base_e100_lr0.1_default')

    # Checkpoint paths
    checkpoint_paths = {
        's0': os.path.join(checkpoint_dir, 's0', 'best.ckpt'),
        's1': os.path.join(checkpoint_dir, 's1', 'best.ckpt'),
        's2': os.path.join(checkpoint_dir, 's2', 'best.ckpt'),
    }

    # Verify checkpoints exist
    for key, path in checkpoint_paths.items():
        if not os.path.isfile(path):
            print(f"Error: Checkpoint not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Build transform
    transform = get_test_transform()

    # ID dataset: CIFAR-10 test
    id_imglist = os.path.join(imglist_dir, 'cifar10', 'test_cifar10.txt')
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=data_dir,
        num_classes=10,
        preprocessor=transform,
        data_aux_preprocessor=None,
    )

    # OOD datasets
    ood_configs = {
        'cifar100': {
            'imglist': os.path.join(imglist_dir, 'cifar100', 'test_cifar100.txt'),
            'num_classes': 100,
        },
        'tin': {
            'imglist': os.path.join(imglist_dir, 'tin', 'test_tin.txt'),
            'num_classes': 200,
        },
    }

    ood_datasets = {}
    for name, cfg in ood_configs.items():
        ood_datasets[name] = ImglistDataset(
            name=f'{name}_test',
            imglist_pth=cfg['imglist'],
            data_dir=data_dir,
            num_classes=cfg['num_classes'],
            preprocessor=transform,
            data_aux_preprocessor=None,
        )

    # DataLoaders
    batch_size = 200
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_loaders = {}
    for name, ds in ood_datasets.items():
        ood_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cpu')

    # Evaluate each checkpoint
    run_metrics = {}
    for run_name, ckpt_path in checkpoint_paths.items():
        # Load model
        model = ResNet18_32x32(num_classes=10)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # Handle potential 'net.' prefix in state dict keys
        if any(k.startswith('net.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.to(device)

        results = evaluate_checkpoint(model, id_loader, ood_loaders, device)
        run_metrics[run_name] = results

    # Compute dataset mean per run, then mean of runs
    dataset_names = list(ood_configs.keys())
    dataset_means_per_run = {}
    for run_name, metrics in run_metrics.items():
        dataset_means_per_run[run_name] = np.mean([metrics[ds] for ds in dataset_names])

    actual = np.mean(list(dataset_means_per_run.values()))

    # Build output
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': len(ood_datasets['cifar100']),
            'tin': len(ood_datasets['tin']),
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
