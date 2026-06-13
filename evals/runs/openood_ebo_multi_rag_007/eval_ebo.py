#!/usr/bin/env python3
"""CPU-safe EBO OOD evaluation for CIFAR-10 using OpenOOD checkpoints."""

import os
import sys
import json
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvs_trans
from PIL import Image, ImageFile

# Fix truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import OpenOOD's exact ResNet18_32x32 implementation
from openood.networks import ResNet18_32x32

# ---------------------------------------------------------------------------
# EBO postprocessor (from openood/postprocessors/ebo_postprocessor.py)
# ---------------------------------------------------------------------------
class EBOPostprocessor:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, logits):
        # energy score: temperature * logsumexp(logits / temperature)
        conf = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return conf  # higher = more in-distribution


# ---------------------------------------------------------------------------
# Dataset loading (from openood/datasets/imglist_dataset.py)
# ---------------------------------------------------------------------------
class ImglistDataset(torch.utils.data.Dataset):
    def __init__(self, imglist_pth, data_dir, transform):
        with open(imglist_pth) as f:
            self.imglist = f.readlines()
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        line = self.imglist[index].strip()
        tokens = line.split(' ', 1)
        image_name = tokens[0]
        path = os.path.join(self.data_dir, image_name)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Preprocessing (from openood/evaluation_api/preprocessor.py)
# ---------------------------------------------------------------------------
def get_cifar10_test_transform():
    return tvs_trans.Compose([
        tvs_trans.Resize(32, interpolation=tvs_trans.InterpolationMode.BILINEAR),
        tvs_trans.CenterCrop(32),
        tvs_trans.ToTensor(),
        tvs_trans.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616]),
    ])


# ---------------------------------------------------------------------------
# AUROC computation (from openood/evaluators/metrics.py)
# ---------------------------------------------------------------------------
def compute_auroc(id_scores, ood_scores):
    """Compute AUROC given ID and OOD confidence scores."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 0.5
    tpr = np.cumsum(labels_sorted == 1) / pos
    fpr = np.cumsum(labels_sorted == 0) / neg
    # Add endpoints
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    return np.trapz(tpr, fpr)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate_run(model, checkpoint_path, temperature, batch_size, data_dir, imglist_dir):
    """Evaluate a single run and return AUROC for CIFAR-100 and TinyImageNet."""
    device = torch.device('cpu')
    
    # Load model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Preprocessing
    transform = get_cifar10_test_transform()
    
    # ID dataset (CIFAR-10 test)
    id_dataset = ImglistDataset(
        imglist_pth=os.path.join(imglist_dir, 'test_cifar10.txt'),
        data_dir=data_dir,
        transform=transform)
    id_loader = torch.utils.data.DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # OOD datasets
    ood_configs = {
        'cifar100': ('test_cifar100.txt', 'cifar100'),
        'tin': ('test_tin.txt', 'tinyimagenet'),
    }
    
    postprocessor = EBOPostprocessor(temperature=temperature)
    
    # Compute ID scores
    id_scores = []
    with torch.no_grad():
        for batch in id_loader:
            logits = model(batch)
            scores = postprocessor(logits)
            id_scores.extend(scores.cpu().numpy())
    id_scores = np.array(id_scores)
    
    results = {}
    for ood_name, (imglist_file, _) in ood_configs.items():
        ood_dataset = ImglistDataset(
            imglist_pth=os.path.join(imglist_dir, imglist_file),
            data_dir=data_dir,
            transform=transform)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        ood_scores = []
        with torch.no_grad():
            for batch in ood_loader:
                logits = model(batch)
                scores = postprocessor(logits)
                ood_scores.extend(scores.cpu().numpy())
        ood_scores = np.array(ood_scores)
        
        auroc = compute_auroc(id_scores, ood_scores)
        results[ood_name] = auroc
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing s0/, s1/, s2/ subdirectories')
    parser.add_argument('--data-dir', type=str, default='./data/images_classic',
                        help='Directory containing images')
    parser.add_argument('--imglist-dir', type=str, default='./data/benchmark_imglist/cifar10',
                        help='Directory containing imglist files')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='EBO temperature parameter')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Find run directories
    run_dirs = sorted(glob(os.path.join(args.root, 's*')))
    if not run_dirs:
        print(f"Error: No run directories found in {args.root}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize model using OpenOOD's exact implementation
    model = ResNet18_32x32(num_classes=10)
    
    all_results = []
    run_names = []
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_names.append(run_name)
        checkpoint_path = os.path.join(run_dir, 'best.ckpt')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
            continue
        
        results = evaluate_run(
            model=model,
            checkpoint_path=checkpoint_path,
            temperature=args.temperature,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            imglist_dir=args.imglist_dir)
        all_results.append(results)
    
    if not all_results:
        print("Error: No successful evaluations", file=sys.stderr)
        sys.exit(1)
    
    # Compute aggregation: dataset mean within each run, then mean of runs
    # First, compute per-run dataset means
    run_dataset_means = []
    for results in all_results:
        dataset_values = list(results.values())
        run_mean = np.mean(dataset_values)
        run_dataset_means.append(run_mean)
    
    # Then mean of runs
    actual = np.mean(run_dataset_means)
    
    # Build output
    # Count datasets per run
    dataset_counts = {}
    for results in all_results:
        for dataset_name in results:
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
    
    # Build run_metrics
    run_metrics = {}
    for i, run_name in enumerate(run_names):
        if i < len(all_results):
            run_metrics[run_name] = {
                'cifar100': float(all_results[i].get('cifar100', 0)),
                'tin': float(all_results[i].get('tin', 0))
            }
    
    output = {
        'metric': 'near_ood_auroc',
        'actual': float(actual),
        'datasets': {
            'cifar100': dataset_counts.get('cifar100', 0),
            'tin': dataset_counts.get('tin', 0)
        },
        'run_metrics': run_metrics,
        'aggregation': 'dataset_mean_then_run_mean'
    }
    
    print(f'REPRO_RESULT {json.dumps(output)}')


if __name__ == '__main__':
    main()
