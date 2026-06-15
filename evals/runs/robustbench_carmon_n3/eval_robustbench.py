#!/usr/bin/env python3
"""
Reproduce robust accuracy of Carmon2019Unlabeled on CIFAR-10 under Linf threat model
using AutoAttack custom version with ['apgd-ce', 'apgd-dlr'], 1 restart each.

Environment: CPU-only, offline. Uses pre-downloaded checkpoint and CIFAR-10 test data.
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

    device = torch.device('cpu')
    dataset = BenchmarkDataset.cifar_10
    threat_model = ThreatModel.Linf

    # 1. Load model - note: load_model does not accept device parameter
    model = load_model(
        model_name=args.model_name,
        model_dir=args.model_dir,
        dataset=dataset,
        threat_model=threat_model
    )
    model = model.to(device)
    model.eval()

    # 2. Load data with correct preprocessing
    preprocessing = get_preprocessing(dataset, threat_model, args.model_name, None)
    x_test, y_test = load_clean_dataset(
        dataset=dataset,
        n_examples=args.n_examples,
        data_dir=args.data_dir,
        prepr=preprocessing
    )
    x_test, y_test = x_test.to(device), y_test.to(device)

    # 3. Configure AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device=device,
        log_path=None
    )
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    # Set number of restarts for each attack
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

    # 4. Run evaluation
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    # 5. Compute robust accuracy (percentage points)
    robust_acc = clean_accuracy(model, x_adv, y_test, device=device)
    robust_acc_pct = robust_acc * 100.0

    # 6. Print result in required format
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == '__main__':
    main()
