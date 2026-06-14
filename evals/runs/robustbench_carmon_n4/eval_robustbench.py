#!/usr/bin/env python3
"""Evaluate Carmon2019Unlabeled robust accuracy on CIFAR-10 with AutoAttack."""

import argparse
import json
import torch
from robustbench.utils import load_model
from robustbench.data import load_cifar10
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

    # Load model
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset='cifar10',
        threat_model=ThreatModel.Linf,
    )
    model.eval()

    # Load data (no external preprocessing needed; model handles normalization internally)
    x_test, y_test = load_cifar10(
        n_examples=args.n_examples,
        data_dir=args.data_dir,
    )

    # Setup AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device='cpu',
    )
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    adversary.apgd.n_restarts = 1

    # Run evaluation
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    # Compute robust accuracy in percentage points (0-100)
    # x_adv is adversarial images (4D tensor), need to pass through model to get logits
    with torch.no_grad():
        logits = model(x_adv)
    correct = (logits.argmax(1) == y_test).float().mean().item()
    robust_acc_pct = correct * 100.0

    # Print result
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
