#!/usr/bin/env python3
"""
Reproduce official EBO Near-OOD AUROC for CIFAR-10 using OpenOOD ResNet18_32x32
checkpoints (s0, s1, s2) and Near-OOD datasets (CIFAR-100, TinyImageNet).
CPU-only, offline. Prints exactly one REPRO_RESULT JSON line.
"""

import json
import os
import sys
import argparse
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Import only the required direct modules (no evaluators/postprocessors API)
# ---------------------------------------------------------------------------
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 0  # CPU-safe

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Official checkpoint paths (relative to root)
CHECKPOINT_REL = {
    "s0": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt",
    "s1": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/best.ckpt",
    "s2": "cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt",
}

# Near-OOD dataset configs: (name, imglist_rel, data_dir_rel)
NEAR_OOD_DATASETS = OrderedDict([
    ("cifar100", ("cifar100/test_cifar100.txt", "images_classic")),
    ("tin", ("tinyimagenet/test_tinyimagenet.txt", "images_classic")),
])

# ---------------------------------------------------------------------------
# Build the exact test transform from openood/preprocessors/transform.py
# ---------------------------------------------------------------------------
def build_test_transform():
    """Replicate TestStandardPreProcessor transform for CIFAR-10."""
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

# ---------------------------------------------------------------------------
# EBO score: max softmax logit (energy-based)
# ---------------------------------------------------------------------------
def ebo_score(logits: torch.Tensor) -> torch.Tensor:
    """Energy-based OOD score = -logsumexp(logits)."""
    return -torch.logsumexp(logits, dim=1)

# ---------------------------------------------------------------------------
# AUROC calculation (percentage)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending (higher score = more OOD-like)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels_sorted == 1)
    neg = np.sum(labels_sorted == 0)
    if pos == 0 or neg == 0:
        return 50.0
    # Compute TPR and FPR
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # AUROC via trapezoidal rule
    auroc = np.trapz(tpr, fpr)
    return float(auroc * 100.0)

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing results/ and data/")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        sys.exit(f"Error: root directory {root} not found")

    # Resolve paths
    data_dir = os.path.join(root, "data")
    results_dir = os.path.join(root, "results")

    # Build transform
    transform = build_test_transform()

    # Load ID (CIFAR-10 test) dataset
    id_imglist = os.path.join(data_dir, "benchmark_imglist", "cifar10", "test_cifar10.txt")
    id_data_dir = os.path.join(data_dir, "images_classic")
    id_dataset = ImglistDataset(
        name="cifar10_test",
        imglist_pth=id_imglist,
        data_dir=id_data_dir,
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS)

    # Prepare OOD datasets
    ood_datasets = {}
    for ood_name, (imglist_rel, data_dir_rel) in NEAR_OOD_DATASETS.items():
        ood_imglist = os.path.join(data_dir, "benchmark_imglist", imglist_rel)
        ood_data_dir = os.path.join(data_dir, data_dir_rel)
        ood_dataset = ImglistDataset(
            name=f"{ood_name}_test",
            imglist_pth=ood_imglist,
            data_dir=ood_data_dir,
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_datasets[ood_name] = ood_dataset

    # Results storage
    run_metrics = OrderedDict()
    dataset_counts = OrderedDict()

    for run_name in ["s0", "s1", "s2"]:
        ckpt_rel = CHECKPOINT_REL[run_name]
        ckpt_path = os.path.join(results_dir, ckpt_rel)
        if not os.path.isfile(ckpt_path):
            sys.exit(f"Error: checkpoint not found at {ckpt_path}")

        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state = torch.load(ckpt_path, map_location="cpu")
        # Handle potential wrapper keys
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
        # Remove module prefix if present
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        # Extract ID scores
        id_scores_list = []
        with torch.no_grad():
            for batch in id_loader:
                images = batch["data"]
                logits = model(images)
                scores = ebo_score(logits)
                id_scores_list.append(scores.cpu().numpy())
        id_scores = np.concatenate(id_scores_list)

        # Evaluate each OOD dataset
        run_metrics[run_name] = OrderedDict()
        for ood_name, ood_dataset in ood_datasets.items():
            ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
            ood_scores_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    images = batch["data"]
                    logits = model(images)
                    scores = ebo_score(logits)
                    ood_scores_list.append(scores.cpu().numpy())
            ood_scores = np.concatenate(ood_scores_list)
            auroc = compute_auroc(id_scores, ood_scores)
            run_metrics[run_name][ood_name] = auroc
            # Store dataset count (number of OOD samples evaluated)
            if ood_name not in dataset_counts:
                dataset_counts[ood_name] = len(ood_dataset)

    # Compute aggregation: dataset mean within each run, then mean of runs
    run_means = []
    for run_name in ["s0", "s1", "s2"]:
        vals = list(run_metrics[run_name].values())
        run_means.append(np.mean(vals))
    actual = float(np.mean(run_means))

    # Build output
    result = {
        "metric": "near_ood_auroc",
        "actual": actual,
        "datasets": dict(dataset_counts),
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean",
    }

    # Print exactly one JSON line
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
