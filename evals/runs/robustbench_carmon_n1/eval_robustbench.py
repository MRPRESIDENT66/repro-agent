#!/usr/bin/env python3
import argparse
import json
import torch
from robustbench.utils import load_model
from robustbench.data import load_cifar10
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
        threat_model='Linf',
    )
    model.eval()

    # Load data (no external preprocessing needed - model handles normalization internally)
    # Use default transforms_test=PREPROCESSINGS[None] which is transforms.ToTensor()
    x_test, y_test = load_cifar10(
        n_examples=args.n_examples,
        data_dir=args.data_dir,
    )

    # Configure AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=args.epsilon,
        version='custom',
        device='cpu'
    )
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr']
    adversary.apgd.n_restarts = 1

    # Run attack
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

    # Compute robust accuracy
    with torch.no_grad():
        outputs = model(x_adv)
        _, predicted = outputs.max(1)
        correct = predicted.eq(y_test).sum().item()
    robust_acc_pct = 100.0 * correct / args.n_examples

    # Print result line
    result = {
        "metric": "robust_accuracy",
        "actual": robust_acc_pct,
        "num_examples": args.n_examples
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == '__main__':
    main()
