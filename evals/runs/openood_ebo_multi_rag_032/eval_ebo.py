#!/usr/bin/env python3
"""CPU-safe EBO evaluation for OpenOOD ResNet18_32x32 on CIFAR-10.

Evaluates three seeds (s0, s1, s2) on near-OOD datasets CIFAR-100 and
TinyImageNet. Prints a single JSON REPRO_RESULT line.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Direct imports from openood modules (no evaluators/postprocessors packages)
from openood.datasets.imglist_dataset import ImglistDataset
from openood.networks.resnet18_32x32 import ResNet18_32x32

# ---------------------------------------------------------------------------
# Constants from OpenOOD repository
# ---------------------------------------------------------------------------
NORMALIZATION_DICT = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
}

NUM_CLASSES = 10
IMAGE_SIZE = 32
BATCH_SIZE = 200

# Benchmark image list paths (relative to --root data directory)
BENCHMARK_DIR = './data/benchmark_imglist/cifar10'
IMAGES_DIR = './data/images_classic'

ID_TEST_LIST = os.path.join(BENCHMARK_DIR, 'test_cifar10.txt')
OOD_LISTS = {
    'cifar100': os.path.join(BENCHMARK_DIR, 'test_cifar100.txt'),
    'tin': os.path.join(BENCHMARK_DIR, 'test_tin.txt'),
}

# Checkpoint root relative to --root
CHECKPOINT_REL = ''


def build_test_transform():
    """Build the test transform matching OpenOOD's TestStandardPreProcessor
    for CIFAR-10: resize to 32x32 bilinear, ToTensor, normalize."""
    mean, std = NORMALIZATION_DICT['cifar10']
    return T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_model(checkpoint_path, device):
    """Load a ResNet18_32x32 with pretrained weights."""
    model = ResNet18_32x32(num_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle possible 'state_dict' key
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def compute_ebo_scores(logits, temperature=1.0):
    """Compute EBO (energy) scores: temperature * logsumexp(logits / temp)."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC in percentage points (0-100)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    # Higher energy -> more OOD
    auroc = roc_auc_score(labels, scores)
    return auroc * 100.0


def evaluate_seed(seed_dir, device, root):
    """Evaluate one seed (s0/s1/s2) on all near-OOD datasets."""
    checkpoint_path = os.path.join(seed_dir, 'best.ckpt')
    if not os.path.isfile(checkpoint_path):
        print(f'Checkpoint not found: {checkpoint_path}', file=sys.stderr)
        return None

    model = load_model(checkpoint_path, device)
    transform = build_test_transform()

    # ID test loader
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(root, ID_TEST_LIST),
        data_dir=os.path.join(root, IMAGES_DIR),
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_loader = DataLoader(
        id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Collect ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            data = batch['data'].to(device)
            logits = model(data)
            scores = compute_ebo_scores(logits).cpu().numpy()
            id_scores.append(scores)
    id_scores = np.concatenate(id_scores)

    results = {}
    for ood_name, ood_list_rel in OOD_LISTS.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=os.path.join(root, ood_list_rel),
            data_dir=os.path.join(root, IMAGES_DIR),
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_loader = DataLoader(
            ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                data = batch['data'].to(device)
                logits = model(data)
                scores = compute_ebo_scores(logits).cpu().numpy()
                ood_scores.append(scores)
        ood_scores = np.concatenate(ood_scores)

        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.',
                        help='Root directory containing data/ and results/')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    device = torch.device('cpu')

    # Locate checkpoint directories
    checkpoint_root = os.path.join(root, CHECKPOINT_REL)
    seed_dirs = sorted([
        os.path.join(checkpoint_root, d)
        for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d)) and d.startswith('s')
    ])

    if not seed_dirs:
        print(f'No seed directories found in {checkpoint_root}', file=sys.stderr)
        sys.exit(1)

    # Evaluate each seed
    run_metrics = {}
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        results = evaluate_seed(seed_dir, device, root)
        if results is not None:
            run_metrics[seed_name] = results

    if not run_metrics:
        print('No valid checkpoints found', file=sys.stderr)
        sys.exit(1)

    # Compute dataset counts (number of ID samples used)
    transform = build_test_transform()
    id_dataset = ImglistDataset(
        name='cifar10_test',
        imglist_pth=os.path.join(root, ID_TEST_LIST),
        data_dir=os.path.join(root, IMAGES_DIR),
        num_classes=NUM_CLASSES,
        preprocessor=transform,
        data_aux_preprocessor=transform,
    )
    id_count = len(id_dataset)

    ood_counts = {}
    for ood_name, ood_list_rel in OOD_LISTS.items():
        ood_dataset = ImglistDataset(
            name=f'{ood_name}_test',
            imglist_pth=os.path.join(root, ood_list_rel),
            data_dir=os.path.join(root, IMAGES_DIR),
            num_classes=NUM_CLASSES,
            preprocessor=transform,
            data_aux_preprocessor=transform,
        )
        ood_counts[ood_name] = len(ood_dataset)

    # Aggregate: dataset mean within each run, then mean of runs
    # First compute per-run dataset mean
    run_dataset_means = []
    for seed_name, metrics in run_metrics.items():
        dataset_values = list(metrics.values())
        run_mean = np.mean(dataset_values)
        run_dataset_means.append(run_mean)

    actual = float(np.mean(run_dataset_means))

    # Build output
    result = {
        'metric': 'near_ood_auroc',
        'actual': round(actual, 4),
        'datasets': {
            'cifar100': ood_counts['cifar100'],
            'tin': ood_counts['tin'],
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean',
    }

    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
