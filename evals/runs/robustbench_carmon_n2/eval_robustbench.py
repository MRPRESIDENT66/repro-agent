#!/usr/bin/env python3
"""Evaluate Carmon2019Unlabeled robust accuracy on CIFAR-10 under Linf AutoAttack."""

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
        dataset='cifar10',
        threat_model='Linf',
        model_dir=args.model_dir,
    )
    model.eval()

    # 2. Load CIFAR-10 test data (first n_examples)
    prepr = get_preprocessing(
        BenchmarkDataset.cifar_10,
        ThreatModel.Linf,
        args.model_name,
        None,
    )
    x_test, y_test = load_clean_dataset(
        dataset=BenchmarkDataset.cifar_10,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        prepr=prepr,
    )

    # 3. Setup AutoAttack (custom version, only APGD-CE and APGD-DLR)
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device='cpu',
        log_path='./log.txt',
    )
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    adversary.apgd.n_restarts = 1

    # 4. Run attack
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    # 5. Compute robust accuracy
    with torch.no_grad():
        outputs = model(x_adv)
        _, y_pred = outputs.max(1)

    robust_acc_frac = (y_pred == y_test).float().mean().item()
    robust_acc_pct = robust_acc_frac * 100.0

    # 6. Print result in required format
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
