#!/usr/bin/env python3
"""
eval_ebo.py — CPU-safe EBO evaluation for OpenOOD ResNet18_32x32 on CIFAR-10.

Reproduces the official Near-OOD AUROC using EBO (temperature=1) on
CIFAR-100 and TinyImageNet, averaged over seeds s0/s1/s2.

Usage:
    python eval_ebo.py --root /path/to/results/cifar10_resnet18_32x32_base_e100_lr0.1_default
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Direct imports from OpenOOD (no evaluators/postprocessors/evaluation_api)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
BATCH_SIZE = 200
NUM_WORKERS = 0  # CPU-safe

# CIFAR-10 normalization (from openood/preprocessors/transform.py)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# Benchmark image list paths (relative to data_dir)
BENCHMARK_DIR = "./data/benchmark_imglist/cifar10"
DATA_DIR = "./data/images_classic"

# Near-OOD datasets
NEAR_OOD_DATASETS = {
    "cifar100": {
        "imglist": os.path.join(BENCHMARK_DIR, "test_cifar100.txt"),
        "data_dir": DATA_DIR,
    },
    "tin": {
        "imglist": os.path.join(BENCHMARK_DIR, "test_tin.txt"),
        "data_dir": DATA_DIR,
    },
}

# ---------------------------------------------------------------------------
# Transform pipeline (from openood/preprocessors/transform.py)
# ---------------------------------------------------------------------------
def get_test_transform():
    """Return the standard CIFAR-10 test transform."""
    return T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset wrapper for ID and OOD data
# ---------------------------------------------------------------------------
class SimpleImglistDataset(Dataset):
    """Minimal wrapper around ImglistDataset for direct use with DataLoader."""

    def __init__(self, imglist_pth, data_dir, transform):
        self.dataset = ImglistDataset(
            name="eval",
            imglist_pth=imglist_pth,
            data_dir=data_dir,
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.getitem(idx)
        return sample["data"], sample["label"]


# ---------------------------------------------------------------------------
# EBO score computation
# ---------------------------------------------------------------------------
def compute_ebo_scores(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute energy scores: -temperature * logsumexp(logits / temperature)."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------
def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute AUROC in percentage points."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # For EBO: lower energy → ID, higher energy → OOD
    # AUROC expects higher score → positive class (ID)
    # So we negate scores: higher neg_energy → ID
    auroc = roc_auc_score(labels, -scores)
    return auroc * 100.0  # percentage points


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EBO evaluation for OpenOOD")
    parser.add_argument("--root", required=True, help="Path to results directory with s0/s1/s2 subfolders")
    args = parser.parse_args()

    root = args.root

    # Find seed subfolders
    seed_dirs = sorted(glob.glob(os.path.join(root, "s*")))
    if not seed_dirs:
        print(f"ERROR: No seed subfolders found in {root}", file=sys.stderr)
        sys.exit(1)

    # Get transform
    transform = get_test_transform()

    # Load ID dataset (CIFAR-10 test)
    id_imglist = os.path.join(BENCHMARK_DIR, "test_cifar10.txt")
    id_dataset = SimpleImglistDataset(id_imglist, DATA_DIR, transform)
    id_loader = DataLoader(
        id_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Load OOD datasets
    ood_loaders = {}
    ood_sizes = {}
    for name, paths in NEAR_OOD_DATASETS.items():
        ds = SimpleImglistDataset(paths["imglist"], paths["data_dir"], transform)
        ood_loaders[name] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        ood_sizes[name] = len(ds)

    # Store per-seed, per-dataset AUROCs
    run_metrics = {}
    all_run_aurocs = []

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        checkpoint_path = os.path.join(seed_dir, "best.ckpt")

        if not os.path.isfile(checkpoint_path):
            print(f"WARNING: Checkpoint not found at {checkpoint_path}, skipping {seed_name}", file=sys.stderr)
            continue

        # Load model
        model = ResNet18_32x32(num_classes=NUM_CLASSES)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle potential 'net.' prefix in state dict keys
        if any(k.startswith("net.") for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.eval()

        # Compute ID scores
        id_scores = []
        with torch.no_grad():
            for inputs, _ in id_loader:
                logits = model(inputs)
                energy = compute_ebo_scores(logits)
                id_scores.append(energy.numpy())
        id_scores = np.concatenate(id_scores)

        # Compute OOD scores per dataset
        per_dataset_aurocs = {}
        for name, loader in ood_loaders.items():
            ood_scores = []
            with torch.no_grad():
                for inputs, _ in loader:
                    logits = model(inputs)
                    energy = compute_ebo_scores(logits)
                    ood_scores.append(energy.numpy())
            ood_scores = np.concatenate(ood_scores)
            auroc = compute_auroc(id_scores, ood_scores)
            per_dataset_aurocs[name] = round(auroc, 2)

        run_metrics[seed_name] = per_dataset_aurocs
        # Dataset mean for this run
        run_mean = np.mean(list(per_dataset_aurocs.values()))
        all_run_aurocs.append(run_mean)

    # Final aggregation: mean of run means
    final_auroc = np.mean(all_run_aurocs)

    # Build result
    result = {
        "metric": "near_ood_auroc",
        "actual": round(float(final_auroc), 2),
        "datasets": {name: ood_sizes[name] for name in NEAR_OOD_DATASETS},
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean",
    }

    # Print exactly one JSON line
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
