#!/usr/bin/env python3
"""eval_detectors.py — Reproduce top-1 accuracy for resnet18_cifar100 on CIFAR-100 test set."""

import json
import torch
import detectors  # required side-effect: registers the custom model with timm
import timm
from timm.data import create_transform, resolve_data_config
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    # 1. Load model with trained weights
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()

    # 2. Get preprocessing from model config (not assumed)
    # Use resolve_data_config to extract only the keys that create_transform accepts
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg, is_training=False)

    # 3. Load dataset (test split, 10000 examples)
    dataset = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)

    # 4. Apply transform, keep only needed columns
    def preprocess(example):
        example["img"] = transform(example["img"])
        return example

    dataset = dataset.map(preprocess, remove_columns=["coarse_label"])
    dataset.set_format(type="torch", columns=["img", "fine_label"])

    # 5. DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 6. Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"]
            labels = batch["fine_label"]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total

    # 7. Print strict JSON result line
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
