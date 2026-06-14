"""Run one strict-blind Multi-Agent + RAG detectors/timm VGG16-bn CIFAR-10 experiment."""

import os

from agent.multi_rag import run_oracle
from evals.oracles.detectors_timm import make_config

if __name__ == "__main__":
    run_oracle(
        make_config(
            attempt=os.environ.get("DETECTORS_ATTEMPT", "001"),
            model_name="vgg16_bn_cifar10",
            dataset_desc="the CIFAR-10 test set (uoft-cs/cifar10, split='test')",
            num_examples=10000,
            num_classes=10,
            expected=93.37,
            label_hint="image field 'img', gold field 'label' (10 classes)",
            workspace_slug="detectors_vgg16_cifar10",
        ),
        pipeline=os.environ.get("PIPELINE", "full"),
    )
