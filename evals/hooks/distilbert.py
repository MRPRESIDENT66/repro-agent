"""Workspace-only hook for the cached DistilBERT benchmark."""

import json
import shutil
from pathlib import Path

from evals.manifest import OracleHooks, TaskManifest

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{MODEL_NAME}"


def _snapshot() -> Path:
    snapshots = sorted((MODEL_CACHE / "snapshots").glob("*"))
    if not snapshots:
        raise RuntimeError(f"model snapshot not found under {MODEL_CACHE}")
    return snapshots[-1]


def _provision(manifest: TaskManifest, workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    (workdir / ".home").mkdir(parents=True)
    config = json.loads((_snapshot() / "config.json").read_text())
    labels = config.get("id2label", {})
    label_lines = "\n".join(f"- `{key}` -> {value}" for key, value in labels.items())
    card = (
        f"# {MODEL_NAME}\n\n"
        "DistilBERT fine-tuned for binary sentiment classification on SST-2.\n\n"
        f"## Label mapping (id2label)\n\n{label_lines}\n\n"
        "These align with SST-2: 0 = negative, 1 = positive.\n\n"
        f"## Dataset\n\nGLUE / SST-2 validation split, "
        f"{manifest.expected_samples} examples; fields `sentence` and `label`.\n"
    )
    (workdir / "model_card.md").write_text(card)


def binding(_manifest: TaskManifest) -> OracleHooks:
    return OracleHooks(provision_override=_provision)
