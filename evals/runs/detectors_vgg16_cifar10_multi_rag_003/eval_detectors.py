#!/usr/bin/env python3
"""eval_detectors.py: Reproduce vgg16_bn_cifar10 top-1 accuracy on CIFAR-10 test set."""

import json
import torch
import detectors          # registers the model with timm (side-effect)
import timm
from datasets import load_dataset
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform

def main():
    # 1. Load model with trained weights
    model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    model.eval()

    # 2. Build transform from model's pretrained_cfg (not assumed)
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)

    # 3. Load CIFAR-10 test set (10,000 examples)
    dataset = load_dataset("uoft-cs/cifar10", split="test")

    def preprocess(example):
        img = example['img'].convert('RGB')
        example['pixel_values'] = transform(img)
        return example

    dataset = dataset.map(preprocess, remove_columns=['img'])
    dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 4. Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['pixel_values']
            labels = batch['label']
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 2),
        "num_examples": 10000
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
