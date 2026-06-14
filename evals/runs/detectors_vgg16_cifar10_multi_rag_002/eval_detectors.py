#!/usr/bin/env python3
"""eval_detectors.py: Reproduce top-1 accuracy of vgg16_bn_cifar10 on CIFAR-10 test set."""

import json
import torch
import detectors  # noqa: F401 – registers vgg16_bn_cifar10 with timm
import timm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def main():
    # 1. Load model with pretrained weights (side-effect import above required)
    model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    model.eval()

    # 2. Read preprocessing config from model's pretrained_cfg (not assumed)
    cfg = model.pretrained_cfg
    input_size = cfg["input_size"][1]  # 32
    mean = cfg["mean"]
    std = cfg["std"]

    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    # 3. Load CIFAR-10 test set (10,000 examples) from local cache
    #    Use the default cache directory so it works offline.
    dataset = load_dataset("uoft-cs/cifar10", split="test")
    # set_transform receives a batch (dict of lists) when DataLoader fetches multiple items.
    # Apply the transform per-image.
    dataset.set_transform(lambda x: {
        "img": [transform(img) for img in x["img"]],
        "label": x["label"],
    })

    # 4. Evaluate
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["img"]
            labels = batch["label"]
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    # 5. Print strict JSON result line (exactly one, via print)
    result = {
        "metric": "top1_accuracy",
        "actual": round(accuracy, 2),
        "num_examples": total,
    }
    print(f"REPRO_RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
