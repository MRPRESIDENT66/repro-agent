"""The generic condition must not encode known task-specific solutions."""

from agent.generic_prompts import GENERIC_PROMPTS
from agent.contracts import generic_task_context as _generic_task_context


def test_generic_prompts_are_repository_agnostic() -> None:
    combined = "\n".join(
        (
            GENERIC_PROMPTS.navigator,
            GENERIC_PROMPTS.reproducer,
            GENERIC_PROMPTS.critic,
            GENERIC_PROMPTS.reviewer,
            GENERIC_PROMPTS.repair,
        )
    ).lower()

    forbidden_answers = (
        "import detectors",
        "fine_label",
        "tools/test.py",
        "get_preprocessing",
        "carmon2019unlabeled",
        "adversary.apgd",
        "imglistdataset",
        "resnet18_32x32",
        "openood",
        "mmpretrain",
        "distilbert",
        "sst-2",
    )
    assert all(answer not in combined for answer in forbidden_answers)


def test_generic_prompts_require_evidence_and_real_execution() -> None:
    combined = "\n".join(
        (
            GENERIC_PROMPTS.navigator,
            GENERIC_PROMPTS.reproducer,
            GENERIC_PROMPTS.critic,
            GENERIC_PROMPTS.reviewer,
            GENERIC_PROMPTS.repair,
        )
    ).lower()

    assert "repository evidence" in combined
    assert "real evaluation" in combined
    assert "do not guess" in combined
    assert "repro_result" not in combined
    assert "predictions.json" not in combined
    assert "public result artifact" in combined
    assert "submit source code" in combined
    assert "not result-file contents" in combined
    assert "exact working call pattern" in combined
    assert "do not repair an api error by guessing" in combined
    assert "latest execution log" in combined
    assert "public runtime" in combined
    assert "do not fall back to a generic library dataset layout" in combined
    assert "do not re-enter" in combined
    assert "same failing chain" in combined
    assert "runtime_probe" in combined
    assert "do not use it to run the full evaluation" in combined


def test_all_v2_oracles_publish_a_generic_artifact_contract() -> None:
    from evals.oracles.detectors_timm import make_config as detectors_config
    from evals.oracles.distilbert_sst2 import make_config as distilbert_config
    from evals.oracles.mmpretrain_resnet18 import make_config as mmpretrain_config
    from evals.oracles.openood_ebo import make_config as openood_config
    from evals.oracles.robustbench_carmon import make_config as robustbench_config

    configs = [
        distilbert_config("generic-contract-test"),
        mmpretrain_config("generic-contract-test"),
        openood_config("generic-contract-test"),
        robustbench_config("generic-contract-test"),
        detectors_config(
            attempt="generic-contract-test",
            model_name="resnet18_cifar100",
            dataset_desc="CIFAR-100 test",
            num_examples=10000,
            num_classes=100,
            expected=79.26,
            workspace_slug="detectors_resnet18_cifar100",
            gold_labels="cifar100_test_fine_labels.json",
        ),
    ]

    for config in configs:
        context = _generic_task_context(config)
        assert config.public_result_protocol
        assert config.public_execution_command
        assert "predictions.json" in context
        assert config.public_execution_command in context
        assert "REPRO_RESULT" not in context
        assert "recompute_fn" in config.verify_kwargs


def test_oracle_workspaces_are_attempt_scoped() -> None:
    from evals.oracles.distilbert_sst2 import make_config

    first = make_config("attempt_a")
    second = make_config("attempt_b")

    assert first.workdir != second.workdir
    assert first.workdir.name == "attempt_a"
    assert second.workdir.name == "attempt_b"
