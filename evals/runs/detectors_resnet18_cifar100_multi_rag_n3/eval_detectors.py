#!/usr/bin/env python3
"""Evaluate resnet18_cifar100 on CIFAR-100 test set (CPU)."""

import json
import torch
import torchvision.transforms as T
from datasets import load_dataset
import detectors  # noqa: F401 — side-effect registers model with timm
import timm

def main():
    # 1. Load model with trained weights
    model = timm.create_model("resnet18_cifar100", pretrained=True)
    model.eval()

    # 2. Read preprocessing config from model's pretrained_cfg
    cfg = model.pretrained_cfg
    mean = cfg["mean"]
    std = cfg["std"]
    input_size = cfg["input_size"]  # (3, 32, 32)

    # 3. Build transform: resize to 32x32, tensor, normalize
    transform = T.Compose([
        T.Resize(input_size[1:]),  # (32, 32)
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # 4. Load CIFAR-100 test split (10,000 examples) from local cache
    #    The dataset is already cached at ./cifar100_cache from a previous download.
    #    Use the full Hugging Face dataset identifier 'uoft-cs/cifar100' which matches
    #    the cached directory structure (uoft-cs___cifar100).
    dataset = load_dataset("uoft-cs/cifar100", split="test", cache_dir="./cifar100_cache")

    # 5. Batched CPU inference
    batch_size = 128
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            images = [transform(img) for img in batch["img"]]
            inputs = torch.stack(images)  # (B, 3, 32, 32)
            labels = torch.tensor(batch["fine_label"])

            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # 6. Compute top-1 accuracy as percentage
    accuracy = 100.0 * correct / total

    # 7. Output strict JSON result
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")

if __name__ == "__main__":
    main()
