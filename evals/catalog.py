"""Single manifest-first entry point for every benchmark task."""

from pathlib import Path

from evals.hooks import binding_for
from evals.manifest import load_manifest, make_oracle_config

TASK_DIR = Path(__file__).resolve().parent / "tasks"


def manifest_path(task: str) -> Path:
    path = TASK_DIR / f"{task}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in TASK_DIR.glob("*.yaml")))
        raise ValueError(f"unknown task {task!r}; available: {available}")
    return path


def make_config(task: str, attempt: str):
    """Build one OracleConfig from its manifest and optional named hook."""
    path = manifest_path(task)
    manifest = load_manifest(path)
    return make_oracle_config(
        path,
        attempt,
        hooks=binding_for(manifest),
    )
