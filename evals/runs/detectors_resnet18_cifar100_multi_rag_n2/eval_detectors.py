#!/usr/bin/env python3
"""eval_detectors.py — Reproduce resnet18_cifar100 top-1 accuracy on CIFAR-100 test set."""

import json
import torch
import torchvision.transforms as T
from datasets import load_dataset
import detectors  # required side-effect: registers the custom model with timm
import timm


def main():
    # 1. Load model with trained weights
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()

    # 2. Get model-specific preprocessing from pretrained_cfg (not assumed)
    cfg = model.pretrained_cfg
    input_size = cfg['input_size'][1]          # 32
    crop_pct = cfg['crop_pct']                 # 0.875
    resize_size = int(input_size / crop_pct)   # 36
    mean = cfg['mean']                         # (0.5071, 0.4867, 0.4408)
    std = cfg['std']                           # (0.2675, 0.2565, 0.2761)

    transform = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    # 3. Load CIFAR-100 test set (10,000 examples)
    dataset = load_dataset("uoft-cs/cifar100", split="test")

    # 4. Evaluate on CPU
    correct = 0
    total = 0
    for example in dataset:
        img = transform(example['img']).unsqueeze(0)  # add batch dim
        label = example['fine_label']                 # 100 classes, not coarse_label
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
        correct += (pred == label)
        total += 1

    accuracy = 100.0 * correct / total

    # 5. Print exactly one strict-JSON REPRO_RESULT line
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
