"""Cross-task checks for the manifest-first catalog."""

from dataclasses import replace

import pytest

from agent.validation.contracts import generic_task_context
from evals.catalog import TASK_DIR, make_config, manifest_path
from evals.hooks import binding_for
from evals.manifest import ManifestRuntime, OracleHooks, load_manifest

TASKS = {
    "clip_vitb32_cifar10",
    "sentence_transformers_stsb",
    "distilbert_sst2",
    "detectors_resnet18_cifar100",
    "detectors_vgg16_cifar10",
    "mmpretrain_resnet18",
    "robustbench_carmon",
    "openood_ebo",
}


def test_catalog_contains_every_shipped_task() -> None:
    assert {path.stem for path in TASK_DIR.glob("*.yaml")} == TASKS


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_manifest_builds_a_blind_oracle_config(task: str) -> None:
    manifest = load_manifest(manifest_path(task))
    config = make_config(task, "catalog-test")
    context = generic_task_context(config)

    assert config.name == manifest.name
    assert config.expected_num_examples == manifest.expected_samples
    assert config.workdir.name == "catalog-test"
    assert config.public_execution_command in context
    assert manifest.output_file in context
    assert str(config.expected) not in context
    assert manifest.hidden_gold is None or manifest.hidden_gold not in context


def test_scalar_list_profile_recomputes_percentage_accuracy(tmp_path) -> None:
    manifest = replace(
        load_manifest(manifest_path("distilbert_sst2")),
        hidden_gold="gold.json",
        expected_samples=3,
    )
    (tmp_path / "gold.json").write_text("[0, 1, 1]")
    (tmp_path / "predictions.json").write_text("[0, 0, 1]")
    runtime = ManifestRuntime(manifest, tmp_path, "scalar", OracleHooks())

    score, count = runtime.recompute(tmp_path)

    assert score == pytest.approx(200 / 3)
    assert count == 3


def test_manifest_hook_names_resolve_explicitly() -> None:
    hooked = {
        "distilbert_sst2",
        "detectors_resnet18_cifar100",
        "detectors_vgg16_cifar10",
    }
    for task in hooked:
        manifest = load_manifest(manifest_path(task))
        assert manifest.hook is not None
        assert binding_for(manifest) != OracleHooks()
    for task in TASKS - hooked:
        assert load_manifest(manifest_path(task)).hook is None


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        make_config("missing-task", "test")
