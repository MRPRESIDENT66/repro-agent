#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe OpenOOD EBO evaluation for CIFAR-10 ID, CIFAR-100 and
TinyImageNet (resized) Near-OOD.  Uses official checkpoints, ImglistDataset,
and the exact test transform from openood/preprocessors/transform.py.

Prints exactly one REPRO_RESULT JSON line.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1.  Import the model and dataset from direct modules (no evaluation_api etc.)
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

# ---------------------------------------------------------------------------
# 2.  Constants (from openood/preprocessors/transform.py)
# ---------------------------------------------------------------------------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# The test transform for CIFAR-10 (32x32) from TestStandardPreProcessor:
#   Convert('RGB') -> Resize(32, interpolation=BILINEAR) -> CenterCrop(32)
#   -> ToTensor() -> Normalize(mean, std)
# For 32x32 images Resize(32) is a no-op but we keep it for faithfulness.
TEST_TRANSFORM = tvs_trans.Compose([
    tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
    tvs_trans.CenterCrop(32),
    tvs_trans.ToTensor(),
    tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# 3.  Helper: build ImglistDataset with the test transform
# ---------------------------------------------------------------------------
def make_dataset(name, imglist_pth, data_dir, num_classes):
    """Return an ImglistDataset using the test transform for both preprocessor
    and data_aux_preprocessor (the latter is unused but required by the API)."""
    return ImglistDataset(
        name=name,
        imglist_pth=imglist_pth,
        data_dir=data_dir,
        num_classes=num_classes,
        preprocessor=TEST_TRANSFORM,
        data_aux_preprocessor=TEST_TRANSFORM,
    )

# ---------------------------------------------------------------------------
# 4.  EBO score function (temperature = 1)
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy score = T * logsumexp(logits / T).  Higher → more OOD-like."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# 5.  AUROC (percentage)
# ---------------------------------------------------------------------------
def auroc(energy_id: np.ndarray, energy_ood: np.ndarray) -> float:
    """Area Under the ROC curve, returned as percentage (e.g. 94.50)."""
    scores = np.concatenate([energy_id, energy_ood])
    labels = np.concatenate([np.ones_like(energy_id), np.zeros_like(energy_ood)])
    # Sort by descending score (higher energy → more OOD-like)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0  # degenerate case
    # TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUC via trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc * 100.0)

# ---------------------------------------------------------------------------
# 6.  Main evaluation routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to checkpoint root, e.g. '
                             'results/cifar10_resnet18_32x32_base_e100_lr0.1_default')
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size

    # -----------------------------------------------------------------------
    # 6a.  Locate checkpoint subfolders (s0, s1, s2)
    # -----------------------------------------------------------------------
    subfolders = sorted(glob.glob(os.path.join(root, 's*')))
    if len(subfolders) == 0:
        print(f'ERROR: No subfolders found in {root}', file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 6b.  Dataset paths (official OpenOOD layout)
    # -----------------------------------------------------------------------
    # ID: CIFAR-10
    id_imglist = './data/benchmark_imglist/cifar10/train_cifar10.txt'   # we use test split
    # Actually for evaluation we need the test split:
    id_imglist_test = './data/benchmark_imglist/cifar10/test_cifar10.txt'
    id_data_dir = './data/images_classic/'

    # Near-OOD: CIFAR-100 and TinyImageNet (resized)
    ood_cifar100_imglist = './data/benchmark_imglist/cifar10/test_cifar100.txt'
    ood_cifar100_data_dir = './data/images_classic/'

    ood_tin_imglist = './data/benchmark_imglist/cifar10/test_tin.txt'
    ood_tin_data_dir = './data/images_classic/'

    # -----------------------------------------------------------------------
    # 6c.  Build datasets (ID test, OOD)
    # -----------------------------------------------------------------------
    id_dataset = make_dataset('cifar10_test', id_imglist_test, id_data_dir, 10)
    ood_cifar100 = make_dataset('cifar100_test', ood_cifar100_imglist, ood_cifar100_data_dir, 100)
    ood_tin = make_dataset('tin_test', ood_tin_imglist, ood_tin_data_dir, 200)

    # -----------------------------------------------------------------------
    # 6d.  DataLoaders (CPU, no extra workers to keep simple)
    # -----------------------------------------------------------------------
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_cifar100_loader = DataLoader(ood_cifar100, batch_size=batch_size, shuffle=False, num_workers=0)
    ood_tin_loader = DataLoader(ood_tin, batch_size=batch_size, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # 6e.  Per-run evaluation
    # -----------------------------------------------------------------------
    run_metrics = {}   # e.g. {'s0': {'cifar100': auroc, 'tin': auroc}, ...}
    dataset_counts = {'cifar100': len(ood_cifar100), 'tin': len(ood_tin)}

    for subfolder in subfolders:
        run_name = os.path.basename(subfolder)  # 's0', 's1', 's2'
        ckpt_path = os.path.join(subfolder, 'best.ckpt')
        if not os.path.isfile(ckpt_path):
            print(f'WARNING: checkpoint not found: {ckpt_path}', file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=10)
        state = torch.load(ckpt_path, map_location='cpu')
        # The checkpoint may contain 'state_dict' key or be the state_dict itself
        if 'state_dict' in state:
            state = state['state_dict']
        # Remove 'module.' prefix if present (DataParallel)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()

        # -------------------------------------------------------------------
        # 6e-i.  Compute ID energies
        # -------------------------------------------------------------------
        id_energies = []
        with torch.no_grad():
            for batch in id_loader:
                data = batch['data']  # already tensor
                logits = model(data)
                energies = ebo_score(logits)
                id_energies.append(energies.cpu().numpy())
        id_energies = np.concatenate(id_energies)

        # -------------------------------------------------------------------
        # 6e-ii.  Compute OOD energies for each Near-OOD dataset
        # -------------------------------------------------------------------
        ood_results = {}
        for ood_name, ood_loader in [('cifar100', ood_cifar100_loader),
                                      ('tin', ood_tin_loader)]:
            ood_energies = []
            with torch.no_grad():
                for batch in ood_loader:
                    data = batch['data']
                    logits = model(data)
                    energies = ebo_score(logits)
                    ood_energies.append(energies.cpu().numpy())
            ood_energies = np.concatenate(ood_energies)
            ood_results[ood_name] = auroc(id_energies, ood_energies)

        run_metrics[run_name] = ood_results

    # -----------------------------------------------------------------------
    # 6f.  Aggregate: dataset mean within each run, then mean of runs
    # -----------------------------------------------------------------------
    # For each run, compute mean AUROC across the two Near-OOD datasets
    run_means = []
    for run_name, metrics in run_metrics.items():
        run_mean = np.mean([metrics['cifar100'], metrics['tin']])
        run_means.append(run_mean)
    actual = float(np.mean(run_means))

    # -----------------------------------------------------------------------
    # 6g.  Print the required JSON line
    # -----------------------------------------------------------------------
    result = {
        'metric': 'near_ood_auroc',
        'actual': actual,
        'datasets': dataset_counts,
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }
    print('REPRO_RESULT ' + json.dumps(result))

if __name__ == '__main__':
    main()
