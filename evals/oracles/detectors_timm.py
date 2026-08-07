"""Compatibility import for the two manifest-backed detectors tasks."""

from evals.catalog import make_config as _make_config


def make_config(*, attempt: str, model_name: str, **_legacy_arguments):
    tasks = {
        "resnet18_cifar100": "detectors_resnet18_cifar100",
        "vgg16_bn_cifar10": "detectors_vgg16_cifar10",
    }
    try:
        task = tasks[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported detectors model {model_name!r}") from exc
    return _make_config(task, attempt)
