import os
import json
import argparse
import glob
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Direct imports as specified in contract
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets.imglist_dataset import ImglistDataset


def get_test_transform():
    """Implement the small torchvision test transform directly from openood/preprocessors/transform.py"""
    # CIFAR-10 mean and std from repository normalization_dict
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def compute_ebo_score(logits, temperature=1.0):
    """Compute EBO (Energy-Based Outlier) score: E(x) = -T * log(sum(exp(f_i(x)/T)))"""
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy.cpu().numpy()


def evaluate_model_on_dataset(model, dataloader, device):
    """Evaluate model on a dataset and return logits and labels"""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['data']
                labels = batch.get('label', None)
            else:
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            
            images = images.to(device)
            logits = model(images)
            
            all_logits.append(logits.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0) if all_labels else None
    
    return all_logits, all_labels


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC given ID and OOD scores"""
    # Higher scores should indicate OOD (EBO: higher energy = more likely OOD)
    # So we don't need to flip the scores
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    try:
        auroc = roc_auc_score(labels, scores) * 100.0  # Convert to percentage
    except ValueError:
        # Handle edge cases where all scores are the same
        auroc = 50.0
    return auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root directory containing s0, s1, s2 subfolders')
    args = parser.parse_args()
    
    # Set up CPU-only execution
    device = torch.device('cpu')
    torch.set_num_threads(1)
    
    # Define dataset paths relative to OpenOOD structure
    data_root = './data/images_classic'
    
    # ID dataset: CIFAR-10 test
    id_imglist_path = os.path.join('data', 'benchmark_imglist', 'cifar10', 'test_cifar10.txt')
    
    # Near-OOD datasets
    near_ood_datasets = {
        'cifar100': os.path.join('data', 'benchmark_imglist', 'cifar10', 'test_cifar100.txt'),
        'tin': os.path.join('data', 'benchmark_imglist', 'cifar10', 'test_tin.txt')
    }
    
    # Get test transform
    test_transform = get_test_transform()
    
    # Load ID dataset once
    id_dataset = ImglistDataset(
        name='cifar10',
        imglist_pth=id_imglist_path,
        data_dir=data_root,
        num_classes=10,
        preprocessor=test_transform,
        data_aux_preprocessor=test_transform
    )
    id_loader = DataLoader(id_dataset, batch_size=200, shuffle=False, num_workers=0)
    
    # Load Near-OOD datasets
    ood_loaders = {}
    for name, imglist_path in near_ood_datasets.items():
        ood_dataset = ImglistDataset(
            name=name,
            imglist_pth=imglist_path,
            data_dir=data_root,
            num_classes=10,  # Not used for OOD detection
            preprocessor=test_transform,
            data_aux_preprocessor=test_transform
        )
        ood_loaders[name] = DataLoader(ood_dataset, batch_size=200, shuffle=False, num_workers=0)
    
    # Find seed subfolders
    seed_folders = sorted(glob.glob(os.path.join(args.root, 's*')))
    if not seed_folders:
        raise ValueError(f'No subfolders found in {args.root}')
    
    # Store results per run
    run_metrics = {}
    all_dataset_counts = {'cifar100': 0, 'tin': 0}
    
    for seed_folder in seed_folders:
        seed_name = os.path.basename(seed_folder)
        
        # Load model checkpoint
        ckpt_path = os.path.join(seed_folder, 'best.ckpt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        
        model = ResNet18_32x32(num_classes=10)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        
        # Evaluate on ID data
        id_logits, _ = evaluate_model_on_dataset(model, id_loader, device)
        id_ebo_scores = compute_ebo_score(id_logits)
        
        # Evaluate on each Near-OOD dataset
        run_metrics[seed_name] = {}
        for dataset_name, ood_loader in ood_loaders.items():
            ood_logits, _ = evaluate_model_on_dataset(model, ood_loader, device)
            ood_ebo_scores = compute_ebo_score(ood_logits)
            
            # Compute AUROC
            auroc = compute_auroc(id_ebo_scores, ood_ebo_scores)
            run_metrics[seed_name][dataset_name] = round(float(auroc), 2)
            
            # Count samples (only once, since datasets are same across runs)
            if all_dataset_counts[dataset_name] == 0:
                all_dataset_counts[dataset_name] = len(ood_ebo_scores)
    
    # Compute final metric: dataset mean within each run, then mean across runs
    run_means = []
    for seed_name in sorted(run_metrics.keys()):
        dataset_aurocs = [run_metrics[seed_name]['cifar100'], run_metrics[seed_name]['tin']]
        run_mean = sum(dataset_aurocs) / len(dataset_aurocs)
        run_means.append(run_mean)
    
    final_actual = sum(run_means) / len(run_means)
    
    # Prepare result in exact required format
    result = {
        "metric": "near_ood_auroc",
        "actual": round(float(final_actual), 2),
        "datasets": all_dataset_counts,
        "run_metrics": run_metrics,
        "aggregation": "dataset_mean_then_run_mean"
    }
    
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
