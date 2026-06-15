#!/usr/bin/env python3
"""Evaluate Carmon2019Unlabeled robust accuracy on CIFAR-10 under Linf threat model using AutoAttack."""

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

    # 1. Load model
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset='cifar10',
        threat_model=ThreatModel.Linf,
        custom_checkpoint=f'{args.model_dir}/cifar10/Linf/{args.model_name}.pt'
    )
    model.eval()

    # 2. Get preprocessing and load CIFAR-10 test data (first n_examples)
    prepr = get_preprocessing(
        dataset=BenchmarkDataset.cifar_10,
        threat_model=ThreatModel.Linf,
        model_name=args.model_name,
        preprocessing=None
    )
    x_test, y_test = load_clean_dataset(
        dataset=BenchmarkDataset.cifar_10,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        prepr=prepr
    )

    # 3. Setup AutoAttack (custom version with apgd-ce and apgd-dlr)
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        attacks_to_run=['apgd-ce', 'apgd-dlr'],
        device='cpu',
        log_path=None
    )
    # Set n_restarts = 1 for APGD (both variants share the same APGDAttack instance)
    adversary.apgd.n_restarts = 1

    # 4. Run attack and compute robust accuracy
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    robust_acc = (x_adv.argmax(1) == y_test).float().mean().item()

    # Convert fraction to percentage points (0–100)
    robust_acc_pct = robust_acc * 100.0

    # 5. Print result in required format
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
