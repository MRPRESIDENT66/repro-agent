"""Declarative task manifests for standard blind-reproduction oracles."""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.types import OracleConfig
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
    repository_path: str
    repository_commit: str
    repository_version: str | None
    repository_exclude: tuple[str, ...]
    model_source: str
    model_mount: str
    model_required: tuple[str, ...]
    dataset_source: str
    dataset_mount: str
    dataset_format: str
    dataset_public_fields: tuple[str, ...]
    expected_samples: int
    task_description: str
    output_file: str
    output_format: str
    output_fields: dict[str, FieldSpec]
    allow_extra_fields: bool
    generated_script: str
    execution_command: str
    execution_timeout: int
    syntax_check: bool
    python_path: str
    environment: dict[str, str]
    session_type: str
    execution_backend: str
    metric: str
    prediction_field: str
    hidden_gold: str
    expected: float
    tolerance: float
    chance_level: float | None

    @property
    def public_result_protocol(self) -> str:
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
            f"The eval must WRITE {self.output_file} as a JSON list of exactly "
            f"{self.expected_samples} objects in input order. Every object must have "
            f"{exact}these fields: {', '.join(fields)}. Values must come from real "
            "evaluation. The verifier ignores printed aggregate metrics and "
            "recomputes the metric from this file and private gold labels."
        )


ProvisionHook = Callable[[TaskManifest, Path], None]
VerifierHook = Callable[[TaskManifest, Path], tuple[float, int] | None]


@dataclass(frozen=True)
class OracleHooks:
    """Optional task code for behavior that a standard manifest cannot express."""

    provision: ProvisionHook | None = None
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

    repository = _section(raw, "repository")
    model = _section(raw, "model")
    dataset = _section(raw, "dataset")
    task = _section(raw, "task")
    output = _section(raw, "output")
    execution = _section(raw, "execution")
    verification = _section(raw, "verification")
    fields = _required(output, "fields", dict)

    manifest = TaskManifest(
        name=_required(raw, "name", str),
        repository_path=_required(repository, "path", str),
        repository_commit=_required(repository, "commit", str),
        repository_version=repository.get("version"),
        repository_exclude=tuple(repository.get("exclude", (".git", "__pycache__", "*.pyc"))),
        model_source=_required(model, "source", str),
        model_mount=_required(model, "mount_as", str),
        model_required=tuple(model.get("required", ())),
        dataset_source=_required(dataset, "public_input", str),
        dataset_mount=_required(dataset, "mount_as", str),
        dataset_format=str(dataset.get("format", "jsonl")),
        dataset_public_fields=tuple(dataset.get("public_fields", ())),
        expected_samples=_required(dataset, "expected_samples", int),
        task_description=_required(task, "description", str).strip(),
        output_file=_required(output, "file", str),
        output_format=_required(output, "format", str),
        output_fields={name: _field_spec(name, spec) for name, spec in fields.items()},
        allow_extra_fields=bool(output.get("allow_extra_fields", False)),
        generated_script=_required(execution, "generated_script", str),
        execution_command=_required(execution, "command", str),
        execution_timeout=_required(execution, "timeout", int),
        syntax_check=bool(execution.get("syntax_check", True)),
        python_path=_required(execution, "python", str),
        environment={str(k): str(v) for k, v in execution.get("environment", {}).items()},
        session_type=str(execution.get("runtime", "local")),
        execution_backend=str(execution.get("backend", "local")),
        metric=_required(verification, "metric", str),
        prediction_field=_required(verification, "prediction_field", str),
        hidden_gold=_required(verification, "hidden_gold", str),
        expected=float(_required(verification, "expected", (int, float))),
        tolerance=float(_required(verification, "tolerance", (int, float))),
        chance_level=(
            float(verification["chance_level"])
            if verification.get("chance_level") is not None
            else None
        ),
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: TaskManifest) -> None:
    if manifest.expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if manifest.output_format != "json":
        raise ValueError("the generic manifest runtime currently supports JSON output")
    if manifest.dataset_format != "jsonl":
        raise ValueError("the generic manifest runtime currently supports JSONL input")
    if manifest.session_type != "local":
        raise ValueError("non-local execution requires a custom Oracle adapter")
    if manifest.prediction_field not in manifest.output_fields:
        raise ValueError("prediction_field must appear in output.fields")
    prediction_spec = manifest.output_fields[manifest.prediction_field]
    if prediction_spec.kind not in {"integer", "number"}:
        raise ValueError("prediction_field must be numeric")
    if manifest.tolerance < 0:
        raise ValueError("verification tolerance cannot be negative")
    for label, path in (
        ("dataset.mount_as", manifest.dataset_mount),
        ("model.mount_as", manifest.model_mount),
        ("output.file", manifest.output_file),
        ("execution.generated_script", manifest.generated_script),
    ):
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


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(gold: list[float], predictions: list[float]) -> float:
    """Dependency-free Spearman correlation with average ranks for ties."""
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("Spearman inputs must have the same non-trivial length")
    left = _average_ranks(gold)
    right = _average_ranks(predictions)
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Spearman input is constant")
    return numerator / (left_norm * right_norm)


def pearson(gold: list[float], predictions: list[float]) -> float:
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("Pearson inputs must have the same non-trivial length")
    gold_mean = math.fsum(gold) / len(gold)
    prediction_mean = math.fsum(predictions) / len(predictions)
    numerator = math.fsum(
        (x - gold_mean) * (y - prediction_mean)
        for x, y in zip(gold, predictions)
    )
    gold_norm = math.sqrt(math.fsum((value - gold_mean) ** 2 for value in gold))
    prediction_norm = math.sqrt(
        math.fsum((value - prediction_mean) ** 2 for value in predictions)
    )
    if gold_norm == 0.0 or prediction_norm == 0.0:
        raise ValueError("Pearson input is constant")
    return numerator / (gold_norm * prediction_norm)


def accuracy(gold: list[float], predictions: list[float]) -> float:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("accuracy inputs must have the same non-zero length")
    return sum(expected == actual for expected, actual in zip(gold, predictions)) / len(gold)


def auroc(gold: list[float], predictions: list[float]) -> float:
    """Binary AUROC where larger prediction scores indicate the positive class."""
    if len(gold) != len(predictions) or len(gold) < 2:
        raise ValueError("AUROC inputs must have the same non-trivial length")
    if any(label not in (0.0, 1.0) for label in gold):
        raise ValueError("AUROC gold labels must be binary")
    positives = sum(label == 1.0 for label in gold)
    negatives = len(gold) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    ranks = _average_ranks(predictions)
    positive_rank_sum = math.fsum(
        rank for rank, label in zip(ranks, gold) if label == 1.0
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


MetricFn = Callable[[list[float], list[float]], float]
METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy,
    "auroc": auroc,
    "pearson": pearson,
    "spearman": spearman,
}


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
        self.workdir = root / "workspaces" / manifest.name / attempt
        self.artifact_dir = root / "evals" / "runs" / f"{manifest.name}_{attempt}"

    def _check_assets(self) -> None:
        manifest = self.manifest
        repository = _resolve(self.root, manifest.repository_path)
        dataset = _resolve(self.root, manifest.dataset_source)
        model = _resolve(self.root, manifest.model_source)
        gold = _resolve(self.root, manifest.hidden_gold)
        missing = [
            str(path)
            for path in (repository, dataset, model, gold)
            if not path.exists()
        ]
        missing.extend(
            str(model / filename)
            for filename in manifest.model_required
            if not (model / filename).exists()
        )
        if missing:
            raise RuntimeError("missing manifest assets: " + ", ".join(missing))
        actual_commit = _repository_head(repository)
        if actual_commit != manifest.repository_commit:
            raise RuntimeError(
                f"repository commit mismatch: {actual_commit} != "
                f"{manifest.repository_commit}"
            )

    def _check_public_dataset(self) -> None:
        manifest = self.manifest
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
        repository = _resolve(self.root, manifest.repository_path)
        dataset = _resolve(self.root, manifest.dataset_source)
        model = _resolve(self.root, manifest.model_source)

        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.copytree(
            repository,
            self.workdir,
            ignore=shutil.ignore_patterns(*manifest.repository_exclude),
        )
        dataset_mount = self.workdir / manifest.dataset_mount
        dataset_mount.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset, dataset_mount)
        model_mount = self.workdir / manifest.model_mount
        model_mount.parent.mkdir(parents=True, exist_ok=True)
        model_mount.symlink_to(model, target_is_directory=True)
        (self.workdir / ".home").mkdir()
        if self.hooks.provision is not None:
            self.hooks.provision(manifest, self.workdir)

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
        gold = _resolve(self.root, self.manifest.hidden_gold).resolve()
        target_markers = (
            f"{self.manifest.expected:.9f}",
            f"{100 * self.manifest.expected:.6f}",
        )
        for path in self.workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.resolve() == gold:
                raise RuntimeError(f"private gold leaked into blind workspace: {path}")
            if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".rst"}:
                continue
            text = path.read_text(errors="replace")
            if any(marker in text for marker in target_markers):
                raise RuntimeError(f"private target leaked into blind workspace: {path}")

    def make_session(self) -> Session:
        manifest = self.manifest
        environment = {
            key: value.replace("{workdir}", str(self.workdir))
            for key, value in manifest.environment.items()
        }
        return Session(
            self.workdir,
            venv_python=_resolve(self.root, manifest.python_path),
            default_timeout=manifest.execution_timeout,
            extra_env=environment,
        )

    def execute(self, session: Session) -> RunResult:
        manifest = self.manifest
        if manifest.syntax_check:
            script = shlex.quote(manifest.generated_script)
            syntax = session.shell(f"python -m py_compile {script}", timeout=60)
            if not syntax.ok:
                return syntax
        return session.shell(
            manifest.execution_command,
            timeout=manifest.execution_timeout,
        )

    def recompute(self, workdir: Path) -> tuple[float, int] | None:
        if self.hooks.verifier is not None:
            return self.hooks.verifier(self.manifest, workdir)
        predictions = _prediction_values(self.manifest, workdir)
        if predictions is None:
            return None
        try:
            gold_path = _resolve(self.root, self.manifest.hidden_gold)
            gold = [float(value) for value in json.loads(gold_path.read_text())]
            if len(gold) != self.manifest.expected_samples:
                return None
            score = METRICS[self.manifest.metric](gold, predictions)
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
    if manifest.metric not in METRICS and hooks.verifier is None:
        raise ValueError(
            f"unknown metric {manifest.metric!r}; use one of {tuple(METRICS)} "
            "or provide a verifier hook"
        )
    runtime = ManifestRuntime(manifest, root.resolve(), attempt, hooks)
    search_exclude = {
        manifest.generated_script,
        manifest.output_file,
        "navigator_report.md",
        "review_report.md",
        "reproducer_public_log.txt",
    }
    return OracleConfig(
        name=manifest.name,
        task=manifest.task_description,
        metric=manifest.metric,
        expected=manifest.expected,
        tolerance=manifest.tolerance,
        attempt=attempt,
        expected_num_examples=manifest.expected_samples,
        recompute_fn=runtime.recompute,
        public_result_protocol=manifest.public_result_protocol,
        public_execution_command=manifest.execution_command,
        workdir=runtime.workdir,
        artifact_dir=runtime.artifact_dir,
        eval_script=manifest.generated_script,
        make_session=runtime.make_session,
        copy_clean_source=runtime.provision,
        execute_eval=runtime.execute,
        execution_backend=manifest.execution_backend,
        chance_level=manifest.chance_level,
        search_extra_exclude=search_exclude,
        assert_blind_workspace=runtime.assert_blind,
    )
