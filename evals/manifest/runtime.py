"""Workspace provisioning, blind checks, execution, and metric recomputation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from evals.assets import check_assets, provision_assets
from evals.execution import (
    check_profile,
    execute as execute_profile,
    make_session as make_profile_session,
    select_profile,
)
from evals.manifest.schema import OracleHooks, TaskManifest
from evals.manifest.verifier import (
    public_check as check_public_output,
    public_diagnostics as diagnose_public_output,
    recompute as recompute_metric,
)
from exec.session import RunResult, Session


def resolve(root: Path, raw: str) -> Path:
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


class ManifestRuntime:
    """Turn one validated manifest into executable Oracle callables."""

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
            resolve(self.root, manifest.repository_path)
            if manifest.repository_path
            else None
        )
        dataset = (
            resolve(self.root, manifest.dataset_source)
            if manifest.dataset_source
            else None
        )
        model = (
            resolve(self.root, manifest.model_source)
            if manifest.model_source
            else None
        )
        gold = (
            resolve(self.root, manifest.hidden_gold)
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
        path = resolve(self.root, manifest.dataset_source)
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
            self._finish_provisioning()
            return
        if manifest.assets:
            provision_assets(self.root, self.workdir, manifest.assets)
            (self.workdir / ".home").mkdir(exist_ok=True)
            if self.hooks.provision is not None:
                self.hooks.provision(manifest, self.workdir)
            self._finish_provisioning()
            return

        repository = (
            resolve(self.root, manifest.repository_path)
            if manifest.repository_path
            else None
        )
        dataset = (
            resolve(self.root, manifest.dataset_source)
            if manifest.dataset_source
            else None
        )
        model = (
            resolve(self.root, manifest.model_source)
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
        self._finish_provisioning()

    def _finish_provisioning(self) -> None:
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
            resolve(self.root, self.manifest.hidden_gold).resolve()
            if self.manifest.hidden_gold
            else None
        )
        forbidden = set(self.manifest.privacy_forbidden_names)
        present = {path.name for path in self.workdir.rglob("*") if path.is_file()}
        leaked = forbidden & present
        if leaked:
            raise RuntimeError(f"private files leaked into blind workspace: {leaked}")
        target_markers = _target_markers(self.manifest)
        text_suffixes = {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".rst",
            ".yaml",
            ".yml",
            ".csv",
            ".sh",
        }
        for path in self.workdir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if gold is not None and path.resolve() == gold:
                raise RuntimeError(f"private gold leaked into blind workspace: {path}")
            if path.suffix.lower() not in text_suffixes:
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
        return check_public_output(self.manifest, self.hooks, workdir)

    def public_diagnostics(self, workdir: Path) -> list[str]:
        return diagnose_public_output(self.manifest, self.hooks, workdir)

    def recompute(self, workdir: Path) -> tuple[float, int] | None:
        return recompute_metric(self.manifest, self.hooks, self.root, workdir)
