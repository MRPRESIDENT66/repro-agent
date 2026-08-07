"""Reusable local and Docker execution profiles for manifest tasks."""

import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evals.assets import resolve
from exec.docker_session import DockerSession
from exec.session import RunResult, Session


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    runtime: str
    backend: str
    generated_script: str
    command: str
    timeout: int
    syntax_check: bool
    python: str
    environment: dict[str, str]
    go_offline: bool = False
    image: str | None = None
    memory: str = "2g"
    cpus: float = 2.0
    required: tuple[str, ...] = ()
    task_suffix: str = ""


def parse_execution(execution: dict[str, Any]) -> tuple[
    ExecutionProfile,
    str | None,
    str | None,
    dict[str, ExecutionProfile],
]:
    base = _profile("default", execution)
    raw_profiles = execution.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("execution.profiles must be a mapping")
    profiles = {}
    for name, values in raw_profiles.items():
        if not isinstance(values, dict):
            raise ValueError("each execution profile must be a mapping")
        profiles[str(name)] = _profile(str(name), values, base=base)
    selector = execution.get("profile_env")
    default_name = execution.get("default_profile")
    if selector is not None and not isinstance(selector, str):
        raise ValueError("execution.profile_env must be a string")
    if default_name is not None and not isinstance(default_name, str):
        raise ValueError("execution.default_profile must be a string")
    if profiles and default_name not in profiles:
        raise ValueError("execution.default_profile must name a declared profile")
    return base, selector, default_name, profiles


def select_profile(
    base: ExecutionProfile,
    selector_env: str | None,
    default_name: str | None,
    profiles: dict[str, ExecutionProfile],
) -> ExecutionProfile:
    if not profiles:
        return base
    selected = (
        os.getenv(selector_env, default_name).strip().lower()
        if selector_env
        else default_name
    )
    try:
        return profiles[str(selected)]
    except KeyError as exc:
        valid = ", ".join(sorted(profiles))
        label = selector_env or "execution profile"
        raise ValueError(f"{label} must be one of: {valid}") from exc


def check_profile(root: Path, profile: ExecutionProfile) -> None:
    missing = [
        str(resolve(root, path))
        for path in profile.required
        if not resolve(root, path).exists()
    ]
    if missing:
        raise RuntimeError("missing execution profile assets: " + ", ".join(missing))
    if profile.runtime == "docker" and not profile.image:
        raise ValueError("Docker execution profile requires image")
    if profile.runtime not in {"local", "docker"}:
        raise ValueError("execution runtime must be local or docker")


def make_session(profile: ExecutionProfile, root: Path, workdir: Path):
    if profile.runtime == "docker":
        return DockerSession(
            workdir,
            image=profile.image or "",
            mem=profile.memory,
            cpus=profile.cpus,
            default_timeout=profile.timeout,
        )
    environment = {
        key: _expand(value, root, workdir)
        for key, value in profile.environment.items()
    }
    return Session(
        workdir,
        venv_python=resolve(root, profile.python),
        default_timeout=profile.timeout,
        extra_env=environment,
    )


def execute(profile: ExecutionProfile, session) -> RunResult:
    if profile.syntax_check:
        script = shlex.quote(profile.generated_script)
        syntax = session.shell(f"python -m py_compile {script}", timeout=120)
        if not syntax.ok:
            return syntax
    return session.shell(profile.command, timeout=profile.timeout)


def _profile(
    name: str,
    values: dict[str, Any],
    *,
    base: ExecutionProfile | None = None,
) -> ExecutionProfile:
    if base is None:
        required = ("generated_script", "command", "timeout", "python")
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError("execution is missing: " + ", ".join(missing))
        profile = ExecutionProfile(
            name=name,
            runtime=str(values.get("runtime", "local")),
            backend=str(values.get("backend", "local")),
            generated_script=str(values["generated_script"]),
            command=str(values["command"]),
            timeout=int(values["timeout"]),
            syntax_check=bool(values.get("syntax_check", True)),
            python=str(values["python"]),
            environment={
                str(key): str(value)
                for key, value in values.get("environment", {}).items()
            },
        )
    else:
        profile = replace(base, name=name)
    updates = {
        "runtime": str(values.get("runtime", profile.runtime)),
        "backend": str(values.get("backend", profile.backend)),
        "command": str(values.get("command", profile.command)),
        "timeout": int(values.get("timeout", profile.timeout)),
        "syntax_check": bool(values.get("syntax_check", profile.syntax_check)),
        "python": str(values.get("python", profile.python)),
        "environment": {
            str(key): str(value)
            for key, value in values.get("environment", profile.environment).items()
        },
        "go_offline": bool(values.get("go_offline", profile.go_offline)),
        "image": str(values["image"]) if values.get("image") else profile.image,
        "memory": str(values.get("memory", profile.memory)),
        "cpus": float(values.get("cpus", profile.cpus)),
        "required": tuple(
            str(value) for value in values.get("required", profile.required)
        ),
        "task_suffix": str(values.get("task_suffix", profile.task_suffix)),
    }
    return replace(profile, **updates)


def _expand(value: str, root: Path, workdir: Path) -> str:
    return value.replace("{root}", str(root)).replace("{workdir}", str(workdir))
