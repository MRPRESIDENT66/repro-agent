from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent.contracts import generic_task_context, make_generic_code_validator
from evals import manifest as framework
from evals.manifest import (
    ManifestRuntime,
    OracleHooks,
    load_manifest,
)
from evals.metrics import accuracy, auroc, pearson, spearman
from evals.oracles import sentence_transformers_stsb as oracle


def _write_predictions(path, similarities, *, reverse_ids: bool = False) -> None:
    ids = list(range(len(similarities)))
    if reverse_ids:
        ids.reverse()
    rows = [
        {"id": identifier, "similarity": value}
        for identifier, value in zip(ids, similarities)
    ]
    (path / "predictions.json").write_text(json.dumps(rows))


def _small_manifest(tmp_path):
    manifest = load_manifest(oracle.MANIFEST)
    return replace(
        manifest,
        repository_path="repo",
        repository_commit="test-commit",
        model_source="model",
        dataset_source="pairs.jsonl",
        hidden_gold="gold.json",
        expected_samples=3,
    )


def test_spearman_handles_ties() -> None:
    assert spearman(
        [0.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 3.0]
    ) == pytest.approx(1.0)
    assert spearman([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_common_metric_registry() -> None:
    assert accuracy([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert pearson([0, 1, 2], [0, 2, 4]) == pytest.approx(1.0)


def test_manifest_builds_existing_oracle_contract_without_private_leaks() -> None:
    config = oracle.make_config("contract_test")
    context = generic_task_context(config)

    assert config.name == "sentence_transformers_stsb"
    assert config.expected_num_examples == 1379
    assert config.execution_backend == "local-offline-cpu-or-mps"
    assert "predictions.json" in context
    assert "evals/oracles/gold" not in context
    assert str(config.expected) not in context
    assert str(config.tolerance) not in context


def test_generic_schema_rejects_wrong_ids(tmp_path) -> None:
    manifest = _small_manifest(tmp_path)
    (tmp_path / "gold.json").write_text(json.dumps([0.0, 1.0, 2.0]))
    _write_predictions(tmp_path, [0.1, 0.2, 0.3], reverse_ids=True)
    runtime = ManifestRuntime(manifest, tmp_path, "schema", OracleHooks())

    assert runtime.recompute(tmp_path) is None


def test_generic_recompute_uses_private_gold(tmp_path) -> None:
    manifest = _small_manifest(tmp_path)
    (tmp_path / "gold.json").write_text(json.dumps([0.0, 1.0, 2.0]))
    _write_predictions(tmp_path, [0.1, 0.2, 0.3])
    runtime = ManifestRuntime(manifest, tmp_path, "metric", OracleHooks())

    assert runtime.recompute(tmp_path) == pytest.approx((1.0, 3))


def test_generic_provisioning_copies_public_assets_and_runs_hook(
    tmp_path, monkeypatch
) -> None:
    manifest = _small_manifest(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "library.py").write_text("VALUE = 1\n")
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(
            json.dumps({"id": index, "sentence1": "a", "sentence2": "b"})
            for index in range(3)
        )
        + "\n"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_text("model")
    (tmp_path / "gold.json").write_text(json.dumps([0.0, 1.0, 2.0]))
    monkeypatch.setattr(framework, "_repository_head", lambda _path: "test-commit")

    def provision_hook(_manifest, workdir):
        (workdir / "hook.txt").write_text("ran")

    runtime = ManifestRuntime(
        manifest,
        tmp_path,
        "provision",
        OracleHooks(provision=provision_hook),
    )
    runtime.provision()

    assert (runtime.workdir / "library.py").is_file()
    assert (runtime.workdir / "stsb_pairs.jsonl").is_file()
    assert (runtime.workdir / "model").is_symlink()
    assert (runtime.workdir / "hook.txt").read_text() == "ran"
    assert not (runtime.workdir / "gold.json").exists()


def test_custom_verifier_hook_replaces_generic_metric(tmp_path) -> None:
    manifest = _small_manifest(tmp_path)
    runtime = ManifestRuntime(
        manifest,
        tmp_path,
        "custom",
        OracleHooks(verifier=lambda _manifest, _workdir: (12.5, 3)),
    )

    assert runtime.recompute(tmp_path) == (12.5, 3)


def test_factory_allows_custom_metric_only_with_verifier_hook(
    tmp_path, monkeypatch
) -> None:
    custom = replace(load_manifest(oracle.MANIFEST), metric="custom_metric")
    monkeypatch.setattr(framework, "load_manifest", lambda _path: custom)

    with pytest.raises(ValueError, match="unknown metric"):
        framework.make_oracle_config(oracle.MANIFEST, "no_hook", root=tmp_path)

    config = framework.make_oracle_config(
        oracle.MANIFEST,
        "with_hook",
        root=tmp_path,
        hooks=OracleHooks(verifier=lambda _manifest, _workdir: (7.0, 1379)),
    )
    assert config.recompute_fn(tmp_path) == (7.0, 1379)


def test_manifest_rejects_workspace_path_escape() -> None:
    manifest = replace(load_manifest(oracle.MANIFEST), output_file="../gold.json")

    with pytest.raises(ValueError, match="inside the workspace"):
        framework._validate_manifest(manifest)


def test_cli_output_path_is_valid_without_hardcoded_filename() -> None:
    validator = make_generic_code_validator(oracle.make_config("validator_test"))
    code = """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.output, "w") as handle:
    json.dump([], handle)
"""

    assert validator(code) == code.strip() + "\n"
