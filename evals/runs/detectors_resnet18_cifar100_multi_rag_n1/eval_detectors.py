#!/usr/bin/env python3
"""eval_detectors.py — Reproduce top-1 accuracy for resnet18_cifar100 on CIFAR-100 test set."""

import json
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import detectors  # noqa: F401 — registers resnet18_cifar100 with timm
import timm

def main():
    # 1. Load model with pretrained weights (requires detectors import side-effect)
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()

    # 2. Read normalization from model's pretrained_cfg (not assumed)
    cfg = model.pretrained_cfg
    mean = cfg['mean']
    std = cfg['std']

    # 3. Preprocessing pipeline: CIFAR-100 native size 32x32
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 4. Load CIFAR-100 test set (10000 examples, fine_label for 100 classes)
    dataset = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)

    correct = 0
    total = len(dataset)  # 10000

    # 5. Batched CPU inference
    with torch.no_grad():
        for sample in dataset:
            img = sample['img']
            label = sample['fine_label']  # 100 classes, not coarse_label (20 classes)

            # Preprocess and add batch dimension
            input_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 32, 32]

            # Forward pass
            logits = model(input_tensor)
            pred = logits.argmax(dim=-1).item()

            if pred == label:
                correct += 1

    accuracy = (correct / total) * 100.0

    # 6. Print result as strict JSON (exactly one line)
    result = {
        "metric": "top1_accuracy",
        "actual": accuracy,
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
