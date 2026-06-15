#!/usr/bin/env python3
"""
Evaluate Carmon2019Unlabeled robust accuracy on CIFAR-10 under Linf threat model.
Uses AutoAttack with apgd-ce and apgd-dlr attacks, 1 restart each.
Prints REPRO_RESULT line with robust accuracy in percentage points.
"""

import argparse
import json
import torch
from robustbench.utils import load_model, clean_accuracy
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

    # Load model
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset='cifar10',
        threat_model=ThreatModel.Linf,
    )
    model.eval()

    # Get preprocessing (ToTensor only for CIFAR-10)
    prepr = get_preprocessing(
        BenchmarkDataset.cifar_10,
        ThreatModel.Linf,
        args.model_name,
        None,
    )

    # Load clean dataset
    x_test, y_test = load_clean_dataset(
        BenchmarkDataset.cifar_10,
        args.n_examples,
        args.data_dir,
        prepr,
    )

    # Configure AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        attacks_to_run=['apgd-ce', 'apgd-dlr'],
        device='cpu',
        log_path='./aa_log.txt',
    )
    # Set number of restarts for each attack
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

    # Run attack
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    # Compute robust accuracy (fraction)
    robust_acc_frac = clean_accuracy(model, x_adv, y_test)

    # Convert to percentage points (0–100)
    robust_acc_pct = robust_acc_frac * 100.0

    # Print exactly one line with REPRO_RESULT
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
