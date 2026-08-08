"""Manifest data model and YAML validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from evals.assets import AssetSpec, parse_assets
from evals.execution import ExecutionProfile, parse_execution
from evals.grouped_scores import GroupedScoresSpec, parse_grouped_scores
from exec.session import RunResult


@dataclass(frozen=True)
class FieldSpec:
    kind: str
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    sequence: bool = False


@dataclass(frozen=True)
class TaskManifest:
    name: str
    hook: str | None
    workspace_slug: str
    artifact_slug: str
    assets: tuple[AssetSpec, ...]
    repository_path: str | None
    repository_commit: str | None
    repository_version: str | None
    repository_exclude: tuple[str, ...]
    model_source: str | None
    model_mount: str | None
    model_required: tuple[str, ...]
    dataset_source: str | None
    dataset_mount: str | None
    dataset_format: str
    dataset_public_fields: tuple[str, ...]
    expected_samples: int
    task_description: str
    output_file: str
    output_format: str
    output_structure: str
    output_protocol: str | None
    grouped_scores: GroupedScoresSpec | None
    output_fields: dict[str, FieldSpec]
    allow_extra_fields: bool
    execution_profile: ExecutionProfile
    execution_profile_env: str | None
    execution_default_profile: str | None
    execution_profiles: dict[str, ExecutionProfile]
    metric: str
    prediction_field: str
    hidden_gold: str | None
    gold_limit: int | None
    metric_scale: float
    expected: float
    tolerance: float
    chance_level: float | None
    privacy_forbidden_names: tuple[str, ...]
    privacy_scrub_globs: tuple[str, ...]

    @property
    def public_result_protocol(self) -> str:
        if self.output_protocol:
            return self.output_protocol.strip()
        if self.output_structure in {"custom", "grouped_scores"}:
            raise ValueError(
                f"{self.output_structure} output structure requires output.protocol"
            )
        if self.output_structure == "values":
            spec = self.output_fields[self.prediction_field]
            details = [spec.kind]
            if spec.minimum is not None or spec.maximum is not None:
                details.append(f"range [{spec.minimum}, {spec.maximum}]")
            if spec.description:
                details.append(spec.description)
            return (
                f"The eval must WRITE `{self.output_file}` as a JSON list of exactly "
                f"{self.expected_samples} values in input order. Every value must be "
                f"a {'; '.join(details)}. Values must come from real evaluation. The "
                "verifier ignores printed aggregate metrics and recomputes the metric "
                "from this file and private gold labels."
            )
        fields = []
        for name, spec in self.output_fields.items():
            details = [spec.kind]
            if spec.sequence:
                details.append(f"consecutive ids 0 through {self.expected_samples - 1}")
            if spec.minimum is not None or spec.maximum is not None:
                details.append(f"range [{spec.minimum}, {spec.maximum}]")
            if spec.description:
                details.append(spec.description)
            fields.append(f"`{name}` ({'; '.join(details)})")
        exact = "exactly " if not self.allow_extra_fields else "at least "
        return (
            f"The eval must WRITE `{self.output_file}` as a JSON list of exactly "
            f"{self.expected_samples} objects in input order. Every object must have "
            f"{exact}these fields: {', '.join(fields)}. Values must come from real "
            "evaluation. The verifier ignores printed aggregate metrics and "
            "recomputes the metric from this file and private gold labels."
        )


ProvisionHook = Callable[[TaskManifest, Path], None]
VerifierHook = Callable[[TaskManifest, Path], tuple[float, int] | None]
SessionHook = Callable[[TaskManifest, Path], Any]
ExecuteHook = Callable[[TaskManifest, Any], RunResult]
PublicCheckHook = Callable[[TaskManifest, Path], bool]
PublicDiagnosticsHook = Callable[[TaskManifest, Path], list[str]]
BlindCheckHook = Callable[[TaskManifest, Path], None]


@dataclass(frozen=True)
class OracleHooks:
    """Optional task code for behavior that a standard manifest cannot express."""

    provision: ProvisionHook | None = None
    provision_override: ProvisionHook | None = None
    session: SessionHook | None = None
    execute: ExecuteHook | None = None
    public_check: PublicCheckHook | None = None
    public_diagnostics: PublicDiagnosticsHook | None = None
    blind_check: BlindCheckHook | None = None
    verifier: VerifierHook | None = None


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"manifest section {name!r} must be a mapping")
    return value


def _required(
    data: dict[str, Any],
    name: str,
    expected_type: type | tuple[type, ...],
) -> Any:
    value = data.get(name)
    expected_types = (
        expected_type if isinstance(expected_type, tuple) else (expected_type,)
    )
    if isinstance(value, bool) and any(kind in (int, float) for kind in expected_types):
        raise ValueError(f"manifest field {name!r} has the wrong type")
    if not isinstance(value, expected_type):
        expected_names = "/".join(kind.__name__ for kind in expected_types)
        raise ValueError(f"manifest field {name!r} must be {expected_names}")
    return value


def _field_spec(name: str, value: Any) -> FieldSpec:
    if isinstance(value, str):
        return FieldSpec(kind=value)
    if not isinstance(value, dict):
        raise ValueError(f"output field {name!r} must be a string or mapping")
    kind = _required(value, "type", str)
    return FieldSpec(
        kind=kind,
        description=str(value.get("description", "")),
        minimum=float(value["minimum"]) if value.get("minimum") is not None else None,
        maximum=float(value["maximum"]) if value.get("maximum") is not None else None,
        sequence=bool(value.get("sequence", False)),
    )


def load_manifest(path: str | Path) -> TaskManifest:
    """Load and validate one private Oracle-side YAML manifest."""
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("task manifest must contain a mapping")

    repository = raw.get("repository", {})
    model = raw.get("model", {})
    workspace = raw.get("workspace", {})
    if not isinstance(repository, dict) or not isinstance(model, dict):
        raise ValueError("repository and model sections must be mappings")
    if not isinstance(workspace, dict):
        raise ValueError("workspace section must be a mapping")
    dataset = _section(raw, "dataset")
    task = _section(raw, "task")
    output = _section(raw, "output")
    execution = _section(raw, "execution")
    verification = _section(raw, "verification")
    fields = _required(output, "fields", dict)
    base_profile, profile_env, default_profile, profiles = parse_execution(execution)
    privacy = raw.get("privacy", {})
    if not isinstance(privacy, dict):
        raise ValueError("privacy section must be a mapping")

    name = _required(raw, "name", str)
    workspace_slug = str(workspace.get("slug", name))
    manifest = TaskManifest(
        name=name,
        hook=str(raw["hook"]) if raw.get("hook") is not None else None,
        workspace_slug=workspace_slug,
        artifact_slug=str(workspace.get("artifact_slug", workspace_slug)),
        assets=parse_assets(raw.get("assets")),
        repository_path=str(repository["path"]) if repository.get("path") else None,
        repository_commit=str(repository["commit"]) if repository.get("commit") else None,
        repository_version=repository.get("version"),
        repository_exclude=tuple(
            repository.get("exclude", (".git", "__pycache__", "*.pyc"))
        ),
        model_source=str(model["source"]) if model.get("source") else None,
        model_mount=str(model["mount_as"]) if model.get("mount_as") else None,
        model_required=tuple(model.get("required", ())),
        dataset_source=(
            str(dataset["public_input"]) if dataset.get("public_input") else None
        ),
        dataset_mount=str(dataset["mount_as"]) if dataset.get("mount_as") else None,
        dataset_format=str(dataset.get("format", "none")),
        dataset_public_fields=tuple(dataset.get("public_fields", ())),
        expected_samples=_required(dataset, "expected_samples", int),
        task_description=_required(task, "description", str).strip(),
        output_file=_required(output, "file", str),
        output_format=_required(output, "format", str),
        output_structure=str(output.get("structure", "records")),
        output_protocol=(
            str(output["protocol"]) if output.get("protocol") is not None else None
        ),
        grouped_scores=parse_grouped_scores(output),
        output_fields={name: _field_spec(name, spec) for name, spec in fields.items()},
        allow_extra_fields=bool(output.get("allow_extra_fields", False)),
        execution_profile=base_profile,
        execution_profile_env=profile_env,
        execution_default_profile=default_profile,
        execution_profiles=profiles,
        metric=_required(verification, "metric", str),
        prediction_field=_required(verification, "prediction_field", str),
        hidden_gold=(
            str(verification["hidden_gold"])
            if verification.get("hidden_gold")
            else None
        ),
        gold_limit=(
            int(verification["gold_limit"])
            if verification.get("gold_limit") is not None
            else None
        ),
        metric_scale=float(verification.get("scale", 1.0)),
        expected=float(_required(verification, "expected", (int, float))),
        tolerance=float(_required(verification, "tolerance", (int, float))),
        chance_level=(
            float(verification["chance_level"])
            if verification.get("chance_level") is not None
            else None
        ),
        privacy_forbidden_names=tuple(
            str(value) for value in privacy.get("forbidden_names", ())
        ),
        privacy_scrub_globs=tuple(
            str(value) for value in privacy.get("scrub_globs", ())
        ),
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: TaskManifest) -> None:
    if manifest.expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if manifest.output_format != "json":
        raise ValueError("the generic manifest runtime currently supports JSON output")
    if manifest.output_structure not in {
        "records",
        "values",
        "grouped_scores",
        "custom",
    }:
        raise ValueError(
            "output.structure must be records, values, grouped_scores, or custom"
        )
    if (
        manifest.output_structure in {"custom", "grouped_scores"}
        and not manifest.output_protocol
    ):
        raise ValueError(
            f"{manifest.output_structure} output structure requires output.protocol"
        )
    if (
        manifest.output_structure in {"records", "values"}
        and manifest.prediction_field not in manifest.output_fields
    ):
        raise ValueError("prediction_field must appear in output.fields")
    if manifest.output_structure in {"records", "values"}:
        prediction_spec = manifest.output_fields[manifest.prediction_field]
        if prediction_spec.kind not in {"integer", "number"}:
            raise ValueError("prediction_field must be numeric")
    if manifest.tolerance < 0:
        raise ValueError("verification tolerance cannot be negative")
    if manifest.gold_limit is not None and manifest.gold_limit <= 0:
        raise ValueError("verification.gold_limit must be positive")
    paths = (
        ("dataset.mount_as", manifest.dataset_mount),
        ("model.mount_as", manifest.model_mount),
        ("output.file", manifest.output_file),
        ("execution.generated_script", manifest.execution_profile.generated_script),
    )
    for label, path in paths:
        if path is None:
            continue
        workspace_path = Path(path)
        if workspace_path.is_absolute() or ".." in workspace_path.parts:
            raise ValueError(f"{label} must stay inside the workspace")
    for name, spec in manifest.output_fields.items():
        if spec.kind not in {"integer", "number", "string", "boolean"}:
            raise ValueError(f"unsupported output type for {name!r}: {spec.kind!r}")
        if spec.sequence and spec.kind != "integer":
            raise ValueError(f"sequential output field {name!r} must be an integer")
