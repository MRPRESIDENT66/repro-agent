"""Declarative task manifests for standard blind-reproduction oracles."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.types import OracleConfig
from evals.assets import AssetSpec, check_assets, parse_assets, provision_assets
from evals.execution import (
    ExecutionProfile,
    check_profile,
    execute as execute_profile,
    make_session as make_profile_session,
    parse_execution,
    select_profile,
)
from evals.grouped_scores import (
    GroupedScoresSpec,
    direction_diagnostics,
    grouped_auroc,
    load_grouped_scores,
    parse_grouped_scores,
)
from evals.metrics import METRICS
from exec.session import RunResult, Session

ROOT = Path(__file__).resolve().parents[1]


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
        repository_exclude=tuple(repository.get("exclude", (".git", "__pycache__", "*.pyc"))),
        model_source=str(model["source"]) if model.get("source") else None,
        model_mount=str(model["mount_as"]) if model.get("mount_as") else None,
        model_required=tuple(model.get("required", ())),
        dataset_source=str(dataset["public_input"]) if dataset.get("public_input") else None,
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


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def matches_glob(path: Path, pattern: str) -> bool:
    """Match recursive globs against both root files and nested files."""
    return path.match(pattern) or (
        pattern.startswith("**/") and path.match(pattern.removeprefix("**/"))
    )


def _target_markers(manifest: TaskManifest) -> set[str]:
    return {
        str(manifest.expected),
        f"{manifest.expected:.9f}",
        f"{100 * manifest.expected:.6f}",
    }


def _repository_head(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect repository commit: {repository}") from exc


def _valid_field(value: Any, spec: FieldSpec, expected_index: int) -> bool:
    if spec.kind == "integer":
        valid = type(value) is int
    elif spec.kind == "number":
        valid = not isinstance(value, bool) and isinstance(value, (int, float))
    elif spec.kind == "string":
        valid = isinstance(value, str)
    else:
        valid = type(value) is bool
    if not valid:
        return False
    if spec.sequence and value != expected_index:
        return False
    if spec.kind in {"integer", "number"}:
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        if spec.minimum is not None and numeric < spec.minimum:
            return False
        if spec.maximum is not None and numeric > spec.maximum:
            return False
    return True


def _prediction_values(manifest: TaskManifest, workdir: Path) -> list[float] | None:
    path = workdir / manifest.output_file
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or len(rows) != manifest.expected_samples:
        return None

    if manifest.output_structure == "custom":
        return None
    if manifest.output_structure == "values":
        spec = manifest.output_fields[manifest.prediction_field]
        if any(
            not _valid_field(value, spec, index)
            for index, value in enumerate(rows)
        ):
            return None
        return [float(value) for value in rows]

    expected_fields = set(manifest.output_fields)
    predictions: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        fields = set(row)
        if expected_fields - fields:
            return None
        if not manifest.allow_extra_fields and fields != expected_fields:
            return None
        if any(
            not _valid_field(row[name], spec, index)
            for name, spec in manifest.output_fields.items()
        ):
            return None
        predictions.append(float(row[manifest.prediction_field]))
    return predictions


class ManifestRuntime:
    """Turn one validated manifest into the callables required by OracleConfig."""

    def __init__(
        self,
        manifest: TaskManifest,
        root: Path,
        attempt: str,
        hooks: OracleHooks,
    ) -> None:
        self.manifest = manifest
        self.root = root
        self.hooks = hooks
        self.profile = select_profile(
            manifest.execution_profile,
            manifest.execution_profile_env,
            manifest.execution_default_profile,
            manifest.execution_profiles,
        )
        self.workdir = root / "workspaces" / manifest.workspace_slug / attempt
        self.artifact_dir = (
            root / "evals" / "runs" / f"{manifest.artifact_slug}_{attempt}"
        )

    def _check_assets(self) -> None:
        manifest = self.manifest
        check_assets(self.root, manifest.assets)
        check_profile(self.root, self.profile)
        repository = (
            _resolve(self.root, manifest.repository_path)
            if manifest.repository_path
            else None
        )
        dataset = (
            _resolve(self.root, manifest.dataset_source)
            if manifest.dataset_source
            else None
        )
        model = (
            _resolve(self.root, manifest.model_source)
            if manifest.model_source
            else None
        )
        gold = (
            _resolve(self.root, manifest.hidden_gold)
            if manifest.hidden_gold
            else None
        )
        missing = [
            str(path)
            for path in (repository, dataset, model, gold)
            if path is not None and not path.exists()
        ]
        missing.extend(
            str(model / filename)
            for filename in manifest.model_required
            if model is not None and not (model / filename).exists()
        )
        if missing:
            raise RuntimeError("missing manifest assets: " + ", ".join(missing))
        if repository is None or manifest.repository_commit is None:
            return
        actual_commit = _repository_head(repository)
        if actual_commit != manifest.repository_commit:
            raise RuntimeError(
                f"repository commit mismatch: {actual_commit} != "
                f"{manifest.repository_commit}"
            )

    def _check_public_dataset(self) -> None:
        manifest = self.manifest
        if manifest.dataset_source is None or manifest.dataset_format != "jsonl":
            return
        path = _resolve(self.root, manifest.dataset_source)
        lines = path.read_text(errors="replace").splitlines()
        if len(lines) != manifest.expected_samples:
            raise RuntimeError(
                f"public dataset count mismatch: {len(lines)} != "
                f"{manifest.expected_samples}"
            )
        expected_fields = set(manifest.dataset_public_fields)
        if not expected_fields:
            return
        for line in lines:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise RuntimeError("public dataset contains undeclared fields")

    def provision(self) -> None:
        self._check_assets()
        self._check_public_dataset()
        manifest = self.manifest
        if self.hooks.provision_override is not None:
            self.hooks.provision_override(manifest, self.workdir)
            self._scrub_private_targets()
            self._write_provenance()
            return
        if manifest.assets:
            provision_assets(self.root, self.workdir, manifest.assets)
            (self.workdir / ".home").mkdir(exist_ok=True)
            if self.hooks.provision is not None:
                self.hooks.provision(manifest, self.workdir)
            self._scrub_private_targets()
            self._write_provenance()
            return
        repository = (
            _resolve(self.root, manifest.repository_path)
            if manifest.repository_path
            else None
        )
        dataset = (
            _resolve(self.root, manifest.dataset_source)
            if manifest.dataset_source
            else None
        )
        model = (
            _resolve(self.root, manifest.model_source)
            if manifest.model_source
            else None
        )

        shutil.rmtree(self.workdir, ignore_errors=True)
        if repository is not None:
            shutil.copytree(
                repository,
                self.workdir,
                ignore=shutil.ignore_patterns(*manifest.repository_exclude),
            )
        else:
            self.workdir.mkdir(parents=True)
        if dataset is not None and manifest.dataset_mount is not None:
            dataset_mount = self.workdir / manifest.dataset_mount
            dataset_mount.parent.mkdir(parents=True, exist_ok=True)
            if dataset.is_dir():
                shutil.copytree(dataset, dataset_mount)
            else:
                shutil.copy2(dataset, dataset_mount)
        if model is not None and manifest.model_mount is not None:
            model_mount = self.workdir / manifest.model_mount
            model_mount.parent.mkdir(parents=True, exist_ok=True)
            model_mount.symlink_to(model, target_is_directory=model.is_dir())
        (self.workdir / ".home").mkdir()
        if self.hooks.provision is not None:
            self.hooks.provision(manifest, self.workdir)

        self._scrub_private_targets()
        self._write_provenance()

    def _scrub_private_targets(self) -> None:
        patterns = self.manifest.privacy_scrub_globs
        if not patterns:
            return
        markers = _target_markers(self.manifest)
        for path in self.workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.workdir)
            if not any(matches_glob(relative, pattern) for pattern in patterns):
                continue
            text = path.read_text(errors="replace")
            for marker in markers:
                text = text.replace(marker, "[scrubbed]")
            path.write_text(text)

    def _write_provenance(self) -> None:
        manifest = self.manifest
        provenance = {
            "name": manifest.name,
            "repository_commit": manifest.repository_commit,
            "repository_version": manifest.repository_version,
            "expected_samples": manifest.expected_samples,
        }
        (self.workdir / "benchmark.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

    def assert_blind(self) -> None:
        if self.hooks.blind_check is not None:
            self.hooks.blind_check(self.manifest, self.workdir)
        gold = (
            _resolve(self.root, self.manifest.hidden_gold).resolve()
            if self.manifest.hidden_gold
            else None
        )
        forbidden = set(self.manifest.privacy_forbidden_names)
        present = {path.name for path in self.workdir.rglob("*") if path.is_file()}
        leaked = forbidden & present
        if leaked:
            raise RuntimeError(f"private files leaked into blind workspace: {leaked}")
        target_markers = _target_markers(self.manifest)
        for path in self.workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if gold is not None and path.resolve() == gold:
                raise RuntimeError(f"private gold leaked into blind workspace: {path}")
            if path.suffix.lower() not in {
                ".py", ".md", ".txt", ".json", ".rst", ".yaml", ".yml",
                ".csv", ".sh",
            }:
                continue
            text = path.read_text(errors="replace")
            if any(marker in text for marker in target_markers):
                raise RuntimeError(f"private target leaked into blind workspace: {path}")

    def make_session(self) -> Session:
        if self.hooks.session is not None:
            return self.hooks.session(self.manifest, self.workdir)
        return make_profile_session(self.profile, self.root, self.workdir)

    def execute(self, session: Session) -> RunResult:
        if self.hooks.execute is not None:
            return self.hooks.execute(self.manifest, session)
        return execute_profile(self.profile, session)

    def public_check(self, workdir: Path) -> bool:
        """Validate only the declared output schema; never load hidden gold."""
        if self.hooks.public_check is not None:
            return self.hooks.public_check(self.manifest, workdir)
        if self.manifest.grouped_scores is not None:
            path = workdir / self.manifest.output_file
            return (
                load_grouped_scores(self.manifest.grouped_scores, path) is not None
                and not direction_diagnostics(self.manifest.grouped_scores, path)
            )
        return _prediction_values(self.manifest, workdir) is not None

    def public_diagnostics(self, workdir: Path) -> list[str]:
        if self.hooks.public_diagnostics is None:
            if self.manifest.grouped_scores is None:
                return []
            return direction_diagnostics(
                self.manifest.grouped_scores,
                workdir / self.manifest.output_file,
            )
        return self.hooks.public_diagnostics(self.manifest, workdir)

    def recompute(self, workdir: Path) -> tuple[float, int] | None:
        if self.hooks.verifier is not None:
            return self.hooks.verifier(self.manifest, workdir)
        if self.manifest.metric in {"grouped_auroc", "near_ood_auroc"}:
            if self.manifest.grouped_scores is None:
                return None
            return grouped_auroc(
                self.manifest.grouped_scores,
                workdir / self.manifest.output_file,
            )
        predictions = _prediction_values(self.manifest, workdir)
        if predictions is None or self.manifest.hidden_gold is None:
            return None
        try:
            gold_path = _resolve(self.root, self.manifest.hidden_gold)
            gold = [float(value) for value in json.loads(gold_path.read_text())]
            if self.manifest.gold_limit is not None:
                gold = gold[: self.manifest.gold_limit]
            if len(gold) != self.manifest.expected_samples:
                return None
            score = (
                METRICS[self.manifest.metric](gold, predictions)
                * self.manifest.metric_scale
            )
        except (OSError, TypeError, ValueError):
            return None
        return score, self.manifest.expected_samples


def make_oracle_config(
    manifest_path: str | Path,
    attempt: str,
    *,
    root: Path = ROOT,
    hooks: OracleHooks | None = None,
) -> OracleConfig:
    """Create the existing pipeline contract from a declarative task manifest."""
    manifest = load_manifest(manifest_path)
    hooks = hooks or OracleHooks()
    if (
        manifest.metric not in METRICS
        and manifest.metric not in {"grouped_auroc", "near_ood_auroc"}
        and hooks.verifier is None
    ):
        raise ValueError(
            f"unknown metric {manifest.metric!r}; use one of {tuple(METRICS)} "
            "or provide a verifier hook"
        )
    runtime = ManifestRuntime(manifest, root.resolve(), attempt, hooks)
    search_exclude = {
        runtime.profile.generated_script,
        manifest.output_file,
        "navigator_report.md",
        "audit_report.md",
        "reproducer_public_log.txt",
    }
    return OracleConfig(
        name=manifest.name,
        task=manifest.task_description + runtime.profile.task_suffix,
        metric=manifest.metric,
        expected=manifest.expected,
        tolerance=manifest.tolerance,
        attempt=attempt,
        expected_num_examples=manifest.expected_samples,
        recompute_fn=runtime.recompute,
        public_check_fn=runtime.public_check,
        public_diagnostics_fn=(
            runtime.public_diagnostics
            if hooks.public_diagnostics is not None or manifest.grouped_scores is not None
            else None
        ),
        public_result_protocol=manifest.public_result_protocol,
        public_execution_command=runtime.profile.command,
        workdir=runtime.workdir,
        artifact_dir=runtime.artifact_dir,
        eval_script=runtime.profile.generated_script,
        make_session=runtime.make_session,
        copy_clean_source=runtime.provision,
        execute_eval=runtime.execute,
        session_go_offline=runtime.profile.go_offline,
        execution_backend=runtime.profile.backend,
        chance_level=manifest.chance_level,
        search_extra_exclude=search_exclude,
        assert_blind_workspace=runtime.assert_blind,
    )
