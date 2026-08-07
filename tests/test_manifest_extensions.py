import json
from dataclasses import replace
from pathlib import Path

import pytest

from evals.assets import check_assets, parse_assets, provision_assets
from evals.execution import parse_execution, select_profile
from evals.grouped_scores import (
    direction_diagnostics,
    grouped_auroc,
    load_grouped_scores,
    parse_grouped_scores,
)
from evals.catalog import manifest_path
from evals.manifest import (
    ManifestRuntime,
    OracleHooks,
    load_manifest,
    matches_glob,
)


def test_assets_support_selective_copy_root_merge_and_symlink(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "keep").mkdir(parents=True)
    (source / "keep" / "data.txt").write_text("data")
    (source / "drop.txt").write_text("drop")
    cache = tmp_path / "cache"
    cache.mkdir()

    assets = parse_assets(
        [
            {"source": "source", "mount_as": ".", "include": ["keep"]},
            {"source": "cache", "mount_as": "model", "mode": "symlink"},
        ]
    )
    check_assets(tmp_path, assets)
    workdir = tmp_path / "workspace"
    provision_assets(tmp_path, workdir, assets)

    assert (workdir / "keep" / "data.txt").read_text() == "data"
    assert not (workdir / "drop.txt").exists()
    assert (workdir / "model").is_symlink()


def test_assets_reject_workspace_escape() -> None:
    with pytest.raises(ValueError, match="inside the workspace"):
        parse_assets([{"source": "repo", "mount_as": "../outside"}])


def test_execution_profile_inherits_defaults_and_uses_env(monkeypatch) -> None:
    base, selector, default, profiles = parse_execution(
        {
            "generated_script": "eval.py",
            "command": "python eval.py",
            "timeout": 60,
            "python": ".venv/bin/python",
            "profile_env": "TEST_BACKEND",
            "default_profile": "docker",
            "profiles": {
                "docker": {"runtime": "docker", "image": "test:latest"},
                "mps": {"backend": "mps", "task_suffix": "Use MPS."},
            },
        }
    )
    monkeypatch.setenv("TEST_BACKEND", "mps")

    selected = select_profile(base, selector, default, profiles)

    assert selected.name == "mps"
    assert selected.command == "python eval.py"
    assert selected.backend == "mps"
    assert selected.task_suffix == "Use MPS."


def test_grouped_scores_validate_direction_and_recompute_mean_auroc(
    tmp_path,
) -> None:
    spec = parse_grouped_scores(
        {
            "structure": "grouped_scores",
            "groups": ["s0", "s1"],
            "series": {"id": 2, "ood": 2},
            "positive_series": ["ood"],
            "negative_series": "id",
            "positive_label": "OOD",
            "negative_label": "ID",
        }
    )
    assert spec is not None
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "s0": {"id": [0, 1], "ood": [2, 3]},
                "s1": {"id": [2, 3], "ood": [0, 1]},
            }
        )
    )

    assert load_grouped_scores(spec, path) is not None
    assert grouped_auroc(spec, path) == pytest.approx((50.0, 4))
    assert direction_diagnostics(spec, path) == [
        "Semantically invalid score direction for s1/ood: the public protocol "
        "requires OOD scores HIGHER than ID, but observed means are "
        "positive=0.5, negative=2.5. Correct polarity while preserving formula, "
        "order, and coverage."
    ]


def test_recursive_privacy_glob_includes_workspace_root() -> None:
    assert matches_glob(Path("README.md"), "**/*.md")
    assert matches_glob(Path("docs/README.md"), "**/*.md")


def test_custom_blind_check_cannot_bypass_generic_privacy_check(tmp_path) -> None:
    manifest = replace(
        load_manifest(manifest_path("distilbert_sst2")),
        privacy_forbidden_names=("private.txt",),
    )
    runtime = ManifestRuntime(
        manifest,
        tmp_path,
        "blind-check",
        OracleHooks(blind_check=lambda _manifest, _workdir: None),
    )
    runtime.workdir.mkdir(parents=True)
    (runtime.workdir / "private.txt").write_text("hidden")

    with pytest.raises(RuntimeError, match="private files leaked"):
        runtime.assert_blind()
