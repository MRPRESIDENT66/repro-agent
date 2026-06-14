#!/usr/bin/env python3
"""eval_detectors.py – Evaluate vgg16_bn_cifar10 on CIFAR-10 test set (CPU)."""

import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
import detectors   # registers the custom model into timm's registry
import timm

def main():
    # 1. Load model with trained weights
    model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    model.eval()
    model.to("cpu")

    # 2. Read preprocessing config from model's pretrained_cfg
    cfg = model.pretrained_cfg
    mean = cfg["mean"]
    std = cfg["std"]
    input_size = cfg["input_size"]   # (C, H, W)
    target_size = input_size[1:]     # (H, W)

    # 3. Build transform
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 4. Load dataset
    dataset = load_dataset("uoft-cs/cifar10", split="test")

    # 5. Batched inference
    batch_size = 128
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            images = [transform(img) for img in batch["img"]]
            images = torch.stack(images)  # (B, C, H, W)
            labels = torch.tensor(batch["label"])

            logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    # 6. Output result as strict JSON
    result = {
        "metric": "top1_accuracy",
        "actual": accuracy,
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
