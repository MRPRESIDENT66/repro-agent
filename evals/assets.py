"""Reusable workspace asset provisioning declared by task manifests."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetSpec:
    source: str
    mount_as: str
    mode: str = "copy"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


def parse_assets(raw: Any) -> tuple[AssetSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("assets must be a list")
    assets = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each asset must be a mapping")
        source = item.get("source")
        mount_as = item.get("mount_as")
        if not isinstance(source, str) or not isinstance(mount_as, str):
            raise ValueError("asset source and mount_as must be strings")
        mode = str(item.get("mode", "copy"))
        if mode not in {"copy", "symlink"}:
            raise ValueError("asset mode must be copy or symlink")
        include = tuple(str(value) for value in item.get("include", ()))
        exclude = tuple(str(value) for value in item.get("exclude", ()))
        _validate_relative("asset.mount_as", mount_as, allow_root=True)
        for value in include:
            _validate_relative("asset.include", value)
        if mode == "symlink" and (include or exclude):
            raise ValueError("symlink assets cannot use include or exclude")
        assets.append(AssetSpec(source, mount_as, mode, include, exclude))
    return tuple(assets)


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def check_assets(root: Path, assets: tuple[AssetSpec, ...]) -> None:
    missing = []
    for asset in assets:
        source = resolve(root, asset.source)
        if not source.exists():
            missing.append(str(source))
            continue
        for relative in asset.include:
            if not (source / relative).exists():
                missing.append(str(source / relative))
    if missing:
        raise RuntimeError("missing manifest assets: " + ", ".join(missing))


def provision_assets(
    root: Path,
    workdir: Path,
    assets: tuple[AssetSpec, ...],
) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    for asset in assets:
        source = resolve(root, asset.source)
        target = workdir if asset.mount_as == "." else workdir / asset.mount_as
        if asset.mode == "symlink":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
            continue
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            continue
        if asset.include:
            target.mkdir(parents=True, exist_ok=True)
            for relative in asset.include:
                _copy(source / relative, target / relative)
            continue
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*asset.exclude),
        )


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _validate_relative(label: str, raw: str, *, allow_root: bool = False) -> None:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the workspace")
    if not allow_root and raw in {"", "."}:
        raise ValueError(f"{label} cannot be the workspace root")
