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
from robustbench.utils import load_model
from robustbench.data import load_clean_dataset, get_preprocessing
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

    # Load model with correct signature: model_name, dataset, threat_model, model_dir
    model = load_model(
        model_name=args.model_name,
        dataset=BenchmarkDataset.cifar_10,
        threat_model=ThreatModel.Linf,
        model_dir=args.model_dir,
    )
    model.eval()

    # Get preprocessing BEFORE loading data - fix: add preprocessing=None
    preprocessing = get_preprocessing(
        dataset=BenchmarkDataset.cifar_10,
        threat_model=ThreatModel.Linf,
        model_name=args.model_name,
        preprocessing=None,
    )

    # Load data with preprocessing applied
    x_test, y_test = load_clean_dataset(
        dataset=BenchmarkDataset.cifar_10,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        prepr=preprocessing,
    )

    # Set up AutoAttack with correct parameters
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device=device,
        log_path=None,
    )

    # Configure attacks
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    # Set n_restarts=1 for APGD attacks (correct attribute path)
    adversary.apgd.n_restarts = 1

    # Run evaluation
    x_adv = adversary.run_standard_evaluation(
        x_test,
        y_test,
        bs=args.n_examples,
    )

    # Compute robust accuracy as percentage (0-100)
    with torch.no_grad():
        outputs = model(x_adv)
        _, predicted = outputs.max(1)
        correct = (predicted == y_test).sum().item()
        robust_acc_pct = 100.0 * correct / args.n_examples

    # Print result in required format
    result = {
        'metric': 'robust_accuracy',
        'actual': robust_acc_pct,
        'num_examples': args.n_examples,
    }
    print(f'REPRO_RESULT {json.dumps(result)}')


if __name__ == '__main__':
    main()
