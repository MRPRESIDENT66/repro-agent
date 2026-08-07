"""Card-scrubbing workspace hook shared by the detectors benchmarks."""

import shutil
from pathlib import Path

from evals.assets import resolve
from evals.manifest import OracleHooks, TaskManifest

ROOT = Path(__file__).resolve().parents[2]


def _scrub(text: str, expected: float) -> str:
    targets = (f"{expected / 100:.4f}".rstrip("0"), f"{expected:.2f}".rstrip("0"))
    kept = []
    for line in text.splitlines():
        lower = line.lower()
        if any(target in line for target in targets):
            continue
        if "test accuracy" in lower or "accuracy:" in lower:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _provision(manifest: TaskManifest, workdir: Path) -> None:
    if manifest.model_source is None:
        raise RuntimeError("detectors manifest requires model.source card")
    shutil.rmtree(workdir, ignore_errors=True)
    (workdir / ".home").mkdir(parents=True)
    card = resolve(ROOT, manifest.model_source)
    (workdir / "model_card.md").write_text(
        _scrub(card.read_text(errors="replace"), manifest.expected)
    )


def _blind_check(manifest: TaskManifest, workdir: Path) -> None:
    targets = (f"{manifest.expected:.2f}", f"{manifest.expected / 100:.4f}".rstrip("0"))
    for path in workdir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".md",
            ".py",
            ".txt",
            ".json",
        }:
            continue
        text = path.read_text(errors="replace")
        if any(target in text for target in targets):
            raise RuntimeError(f"private target leaked into blind workspace: {path}")


def binding(_manifest: TaskManifest) -> OracleHooks:
    return OracleHooks(
        provision_override=_provision,
        blind_check=_blind_check,
    )
