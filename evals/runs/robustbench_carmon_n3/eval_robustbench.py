#!/usr/bin/env python3
"""
Reproduce robust accuracy of Carmon2019Unlabeled on CIFAR-10 under Linf threat model
using AutoAttack custom version with ['apgd-ce', 'apgd-dlr'], 1 restart each.

Environment: CPU-only, offline.
Checkpoint: robustbench_models/cifar10/Linf/Carmon2019Unlabeled.pt
Data: robustbench_data (first 50 CIFAR-10 test examples)
Epsilon: 0.031372549
"""

import argparse
import json
import torch
import torchvision.transforms as transforms
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from autoattack import AutoAttack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled')
    parser.add_argument('--model_dir', type=str, default='robustbench_models')
    parser.add_argument('--data_dir', type=str, default='robustbench_data')
    parser.add_argument('--n_examples', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=0.031372549)
    args = parser.parse_args()

    device = torch.device('cpu')
    dataset = BenchmarkDataset.cifar_10
    threat_model = ThreatModel.Linf

    # Load model (no device argument - loads to CPU by default)
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset=dataset,
        threat_model=threat_model,
    )
    model = model.to(device)
    model.eval()

    # Preprocessing: ToTensor only (no normalization for Carmon2019Unlabeled under Linf)
    preprocessing = transforms.Compose([transforms.ToTensor()])

    # Load data
    x_test, y_test = load_clean_dataset(
        dataset=dataset,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        prepr=preprocessing,
    )

    # Set up AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device=device,
        log_path=None,
    )
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    adversary.apgd.n_restarts = 1

    # Run evaluation
    x_adv = adversary.run_standard_evaluation(
        x_test,
        y_test,
        bs=args.n_examples,
    )

    # Compute robust accuracy
    with torch.no_grad():
        logits = model(x_adv)
        preds = logits.argmax(dim=1)
        robust_acc = (preds == y_test).float().mean().item()

    # Print result in required format (percentage points 0-100)
    result = {
        'metric': 'robust_accuracy',
        'actual': round(robust_acc * 100, 2),
        'num_examples': args.n_examples,
    }
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
