"""Run one strict-blind Multi-Agent + RAG detectors/timm ResNet-18 CIFAR-100 experiment."""

import os

from agent.multi_rag import run_oracle
from evals.oracles.detectors_timm import make_config

if __name__ == "__main__":
    run_oracle(
        make_config(
            attempt=os.environ.get("DETECTORS_ATTEMPT", "001"),
            model_name="resnet18_cifar100",
            dataset_desc="the CIFAR-100 test set (uoft-cs/cifar100, split='test')",
            num_examples=10000,
            num_classes=100,
            expected=79.26,
            label_hint="image field 'img', gold field 'fine_label' (100 classes; "
            "the split also has 'coarse_label' with only 20 classes — do not use it)",
            workspace_slug="detectors_resnet18_cifar100",
        )
    )
