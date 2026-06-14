#!/usr/bin/env python3
"""Evaluate resnet18_cifar100 on CIFAR-100 test set (CPU)."""

import json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
import detectors  # noqa: F401  # registers model with timm
import timm

def main():
    # 1. Load model with pretrained weights
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()

    # 2. Get preprocessing config from model's pretrained_cfg
    cfg = model.pretrained_cfg
    mean = cfg['mean']
    std = cfg['std']
    input_size = cfg['input_size']  # (3, H, W)
    interpolation = cfg['interpolation']

    # 3. Build transform: resize to model's input size, convert to tensor, normalize
    transform = T.Compose([
        T.Resize(input_size[1:], interpolation=T.InterpolationMode(interpolation)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # 4. Load CIFAR-100 test set (10000 examples)
    dataset = load_dataset("uoft-cs/cifar100", split="test")
    # Fields: 'img' (PIL image), 'fine_label' (int 0-99), 'coarse_label' (int 0-19)
    # We use 'fine_label' for 100-class classification

    correct = 0
    total = 0

    # 5. Batched CPU inference
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        images = [transform(img) for img in batch['img']]
        labels = torch.tensor(batch['fine_label'], dtype=torch.long)

        # Stack into a batch tensor
        inputs = torch.stack(images, dim=0)

        with torch.no_grad():
            logits = model(inputs)
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total

    # 6. Print result as strict JSON line
    result = {
        "metric": "top1_accuracy",
        "actual": accuracy,
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
