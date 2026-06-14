#!/usr/bin/env python3
"""Reproduce top-1 accuracy of resnet18_cifar100 on CIFAR-100 test set."""

import json
import torch
import timm
from datasets import load_dataset

# Required side-effect import: registers the custom model with timm
import detectors


def main():
    # 1. Load model (CPU only)
    device = torch.device("cpu")
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()
    model.to(device)

    # 2. Get preprocessing config from model's pretrained_cfg
    cfg = model.pretrained_cfg
    data_config = timm.data.resolve_data_config(cfg)
    transform = timm.data.create_transform(**data_config)

    # 3. Load CIFAR-100 test set
    dataset = load_dataset("uoft-cs/cifar100", split="test")

    # 4. Evaluate
    correct = 0
    total = 0
    batch_size = 64  # safe for CPU

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        # Hugging Face datasets slicing returns a dict-like batch where
        # batch['img'] is a list of PIL images and batch['fine_label'] is a list of labels.
        images_raw = batch['img']
        labels_raw = batch['fine_label']

        images = []
        for img in images_raw:
            img_tensor = transform(img).unsqueeze(0)
            images.append(img_tensor)
        images = torch.cat(images, dim=0).to(device)
        labels = torch.tensor(labels_raw, device=device)

        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    # 5. Print result as strict JSON
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 1),
        "num_examples": total
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
