#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO Near-OOD AUROC reproduction for CIFAR-10.

Reproduces official OpenOOD EBO Near-OOD AUROC using s0/s1/s2 checkpoints.
Prints exactly one line: REPRO_RESULT {...}
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

# ---------------------------------------------------------------------------
# Minimal imports from OpenOOD — avoid broad package imports
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset
import numpy as np
from sklearn import metrics


def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)
    accuracy = acc(pred, label)
    results = [fpr, auroc, aupr_in, aupr_out, accuracy]
    return results


def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]
    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)
    return acc


def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


def auc_and_fpr_recall(conf, label, tpr_th):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(1 - ood_indicator, conf)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(ood_indicator, -conf)
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)
    return auroc, aupr_in, aupr_out, fpr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ID_NAME = 'cifar10'
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 0  # CPU-safe
TEMPERATURE = 1.0

# Official checkpoint layout
CHECKPOINT_REL = 'results/cifar10_resnet18_32x32_base_e100_lr0.1_default'
SEEDS = ['s0', 's1', 's2']

# Near-OOD datasets (from official configs/datasets/cifar10/cifar10_ood.yml)
OOD_DATASETS = {
    'cifar100': {
        'imglist': 'data/benchmark_imglist/cifar10/test_cifar100.txt',
        'data_dir': 'data/images_classic/',
    },
    'tin': {
        'imglist': 'data/benchmark_imglist/cifar10/test_tin.txt',
        'data_dir': 'data/images_classic/',
    },
}

# CIFAR-10 normalization (from OpenOOD configs)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_preprocessor(train=False):
    """Return the standard CIFAR-10 test preprocessor."""
    transform_list = [
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(transform_list)


def load_model(checkpoint_path, device='cpu'):
    """Load ResNet18_32x32 with official checkpoint."""
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle possible 'state_dict' key
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def compute_ebo_score(logits, temperature=TEMPERATURE):
    """EBO score: temperature * logsumexp(logits / temperature)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


@torch.no_grad()
def evaluate_seed(model, id_loader, ood_loaders, device='cpu'):
    """Run EBO evaluation for one seed, return per-dataset AUROC."""
    # --- ID inference ---
    id_conf_list = []
    id_pred_list = []
    id_label_list = []
    for batch in id_loader:
        data = batch['data'].to(device)
        labels = batch['label']
        logits = model(data)
        scores = compute_ebo_score(logits)
        _, preds = torch.max(logits, dim=1)
        id_conf_list.append(scores.cpu())
        id_pred_list.append(preds.cpu())
        id_label_list.append(labels)

    id_conf = torch.cat(id_conf_list).numpy()
    id_pred = torch.cat(id_pred_list).numpy()
    id_label = torch.cat(id_label_list).numpy()

    # --- OOD inference per dataset ---
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_conf_list = []
        ood_pred_list = []
        ood_label_list = []
        for batch in ood_loader:
            data = batch['data'].to(device)
            labels = batch['label']
            logits = model(data)
            scores = compute_ebo_score(logits)
            _, preds = torch.max(logits, dim=1)
            ood_conf_list.append(scores.cpu())
            ood_pred_list.append(preds.cpu())
            ood_label_list.append(labels)

        ood_conf = torch.cat(ood_conf_list).numpy()
        ood_pred = torch.cat(ood_pred_list).numpy()
        ood_label = torch.cat(ood_label_list).numpy()

        # Combine ID and OOD
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_label, ood_label])
        pred = np.concatenate([id_pred, ood_pred])

        # Compute metrics (AUROC is second element, already in [0,1])
        metrics = compute_all_metrics(conf, label, pred)
        auroc = metrics[1] * 100.0  # Convert to percentage
        results[ood_name] = auroc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=CHECKPOINT_REL,
                        help='Root directory containing s0/s1/s2 subfolders')
    args = parser.parse_args()

    root = args.root
    device = 'cpu'

    # --- Setup preprocessor ---
    preprocessor = get_preprocessor(train=False)

    # --- ID dataset (CIFAR-10 test) ---
    id_imglist = 'data/benchmark_imglist/cifar10/test.txt'
    id_data_dir = 'data/images_classic/'
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=preprocessor,
        data_aux_preprocessor=preprocessor,
    )
    id_loader = DataLoader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --- OOD datasets ---
    ood_loaders = {}
    for ood_name, ood_info in OOD_DATASETS.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=ood_info['imglist'],
            data_dir=ood_info['data_dir'],
            num_classes=NUM_CLASSES,
            preprocessor=preprocessor,
            data_aux_preprocessor=preprocessor,
        )
        ood_loaders[ood_name] = DataLoader(
            ood_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    # --- Evaluate each seed ---
    run_metrics = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(root, seed, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f'Checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)

        model = load_model(ckpt_path, device=device)
        results = evaluate_seed(model, id_loader, ood_loaders, device=device)
        run_metrics[seed] = results

    # --- Aggregate ---
    # Dataset mean within each run, then mean of runs
    dataset_names = list(OOD_DATASETS.keys())
    dataset_means = {}
    for dname in dataset_names:
        vals = [run_metrics[s][dname] for s in SEEDS]
        dataset_means[dname] = np.mean(vals)

    # Run means (mean across datasets for each run)
    run_means = []
    for s in SEEDS:
        run_vals = [run_metrics[s][d] for d in dataset_names]
        run_means.append(np.mean(run_vals))
    actual = np.mean(run_means)

    # --- Build output ---
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {d: int(len(ood_loaders[d].dataset)) for d in dataset_names},
        'run_metrics': {
            s: {d: float(run_metrics[s][d]) for d in dataset_names}
            for s in SEEDS
        },
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
