#!/usr/bin/env python3
"""
Reproduce official OpenOOD EBO Near-OOD AUROC for CIFAR-10.
CPU-only, offline, using official checkpoints s0/s1/s2.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader

# Import only the required modules; avoid evaluation_api, evaluators, postprocessors.
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_WORKERS = 4

# Official checkpoint paths relative to root
CHECKPOINT_REL = {
    "s0": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt",
    "s1": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt",
    "s2": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt",
}

# Dataset image list paths relative to root
DATASET_CONFIGS = {
    "cifar100": {
        "imglist_pth": "data/cifar10/benchmark_imglist/cifar10/test_cifar100.txt",
        "data_dir": "data/cifar10/benchmark_imglist/cifar10",
    },
    "tin": {
        "imglist_pth": "data/cifar10/benchmark_imglist/cifar10/test_tin.txt",
        "data_dir": "data/cifar10/benchmark_imglist/cifar10",
    },
}

# ---------------------------------------------------------------------------
# Transform: exact reproduction of TestStandardPreProcessor for CIFAR-10
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the exact torchvision test transform used by OpenOOD for CIFAR-10."""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=mean, std=std),
    ])

# ---------------------------------------------------------------------------
# EBO score function
# ---------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy-based score: -logsumexp(logits / T)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points. Higher score = more ID-like."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 50.0
    # Compute TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)  # percentage points

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing data/ and results/")
    args = parser.parse_args()

    root = args.root
    device = torch.device("cpu")

    # Build model
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    model.eval()
    model.to(device)

    # Transform
    transform = get_test_transform()

    # Results storage
    run_metrics = {}
    dataset_counts = {}

    for run_key, ckpt_rel in CHECKPOINT_REL.items():
        ckpt_path = os.path.join(root, ckpt_rel)
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        state = torch.load(ckpt_path, map_location=device)
        # Handle possible wrapping
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif "net" in state:
            model.load_state_dict(state["net"])
        else:
            model.load_state_dict(state)

        model.eval()

        run_metrics[run_key] = {}

        for ds_name, ds_cfg in DATASET_CONFIGS.items():
            imglist_pth = os.path.join(root, ds_cfg["imglist_pth"])
            data_dir = os.path.join(root, ds_cfg["data_dir"])

            if not os.path.isfile(imglist_pth):
                print(f"Image list not found: {imglist_pth}", file=sys.stderr)
                sys.exit(1)

            # Determine if this is ID (CIFAR-10) or OOD
            # For Near-OOD evaluation, we treat CIFAR-100 and TinyImageNet as OOD
            # ID is CIFAR-10, but we only evaluate OOD datasets here.
            # The AUROC is computed between ID (CIFAR-10) and each OOD dataset.
            # We need ID scores for each run. We'll compute them once per run.
            # But to keep structure simple, we compute ID scores per run outside the loop.
            # We'll compute ID scores once per run before the OOD loop.
            pass

        # Compute ID scores for this run
        id_cfg = {
            "imglist_pth": "data/cifar10/benchmark_imglist/cifar10/test_id.txt",
            "data_dir": "data/cifar10/benchmark_imglist/cifar10",
        }
        id_imglist_pth = os.path.join(root, id_cfg["imglist_pth"])
        id_data_dir = os.path.join(root, id_cfg["data_dir"])

        if not os.path.isfile(id_imglist_pth):
            print(f"ID image list not found: {id_imglist_pth}", file=sys.stderr)
            sys.exit(1)

        id_dataset = ImglistDataset(
            name="cifar10_id",
            imglist_pth=id_imglist_pth,
            data_dir=id_data_dir,
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        id_loader = DataLoader(
            id_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch["data"].to(device)
                logits = model(images)
                scores = energy_score(logits)
                id_scores_list.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores_list)

        # Now evaluate each OOD dataset
        for ds_name, ds_cfg in DATASET_CONFIGS.items():
            imglist_pth = os.path.join(root, ds_cfg["imglist_pth"])
            data_dir = os.path.join(root, ds_cfg["data_dir"])

            ood_dataset = ImglistDataset(
                name=ds_name,
                imglist_pth=imglist_pth,
                data_dir=data_dir,
                num_classes=NUM_CLASSES,
                preprocessor=transform,
                data_aux_preprocessor=transform,
            )
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch["data"].to(device)
                    logits = model(images)
                    scores = energy_score(logits)
                    ood_scores_list.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores_list)

            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_key][ds_name] = auroc
            dataset_counts[ds_name] = len(ood_dataset)

    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, for each run, compute mean over datasets
    run_means = []
    for run_key in ["s0", "s1", "s2"]:
        ds_aurocs = [run_metrics[run_key][ds] for ds in ["cifar100", "tin"]]
        run_mean = np.mean(ds_aurocs)
        run_means.append(run_mean)

    actual = float(np.mean(run_means))

    # Build result
    result = {
        "metric": "near_ood_auroc",
        "actual": actual,
        "datasets": dataset_counts,
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean",
    }

    print("REPRO_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
