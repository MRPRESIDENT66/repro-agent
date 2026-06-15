#!/usr/bin/env python3
"""Evaluate Carmon2019Unlabeled robust accuracy on CIFAR-10 under Linf threat model using AutoAttack."""

import argparse
import json
import torch
from robustbench.utils import load_model
from robustbench.data import load_cifar10, get_preprocessing
from autoattack import AutoAttack


def main():
    parser = argparse.ArgumentParser(description='Evaluate robust accuracy of Carmon2019Unlabeled')
    parser.add_argument('--model_name', type=str, default='Carmon2019Unlabeled',
                        help='Model name from RobustBench model zoo')
    parser.add_argument('--model_dir', type=str, default='robustbench_models',
                        help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, default='robustbench_data',
                        help='Directory containing CIFAR-10 dataset')
    parser.add_argument('--n_examples', type=int, default=50,
                        help='Number of test examples to evaluate')
    parser.add_argument('--epsilon', type=float, default=0.031372549,
                        help='Linf perturbation budget')
    args = parser.parse_args()

    # Load model
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset='cifar10',
        threat_model='Linf'
    )
    model.eval()

    # Get preprocessing and load data
    prepr = get_preprocessing('cifar10', 'Linf', args.model_name, None)
    x_test, y_test = load_cifar10(
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        transforms_test=prepr
    )

    # Setup AutoAttack with custom version
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device='cpu',
        log_path=None
    )

    # Set attacks and restarts
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    adversary.apgd.n_restarts = 1

    # Run evaluation
    x_adv = adversary.run_standard_evaluation(
        x_test,
        y_test,
        bs=min(args.n_examples, 50)
    )

    # Compute robust accuracy
    with torch.no_grad():
        logits = model(x_adv)
        preds = logits.argmax(dim=1)
        robust_acc = (preds == y_test).float().mean().item()

    # Convert to percentage points (0-100)
    robust_acc_pct = robust_acc * 100.0

    # Print result in required format
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
