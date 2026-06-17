"""Restricted runtime probes for import, signature, path, and CLI diagnostics."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any


MAX_RUNTIME_PROBES = 8
MAX_RUNTIME_PROBES_PER_ROLE = 2

RUNTIME_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "runtime_probe",
        "description": (
            "Run one restricted diagnostic in the provisioned runtime. "
            "Use it to check an import, inspect a Python signature, list a local "
            "path, or request CLI help. It cannot run arbitrary shell commands or "
            "the full evaluation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "import_smoke",
                        "python_signature",
                        "path_list",
                        "cli_help",
                    ],
                },
                "target": {
                    "type": "string",
                    "description": (
                        "A dotted Python module/object, or a workspace-relative "
                        "path for path_list/cli_help."
                    ),
                },
            },
            "required": ["kind", "target"],
        },
    },
}

_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def runtime_probe_command(kind: str, target: str) -> str:
    """Build a command from a small diagnostic vocabulary, never raw shell."""
    kind = kind.strip()
    target = target.strip()
    if len(target) > 240:
        raise ValueError("runtime probe target is too long")

    if kind in {"import_smoke", "python_signature"}:
        if not _DOTTED_NAME.fullmatch(target):
            raise ValueError("Python probe target must be a dotted identifier")
        if kind == "import_smoke":
            code = (
                "import importlib; "
                f"m=importlib.import_module({target!r}); "
                "print('IMPORTED', m.__name__); "
                "print('FILE', getattr(m, '__file__', None)); "
                "print('VERSION', getattr(m, '__version__', None))"
            )
        else:
            code = f"""import importlib
import inspect
parts = {target!r}.split(".")
obj = None
attrs = []
for index in range(len(parts), 0, -1):
    module_name = ".".join(parts[:index])
    try:
        obj = importlib.import_module(module_name)
        attrs = parts[index:]
        break
    except ModuleNotFoundError as exc:
        if exc.name != module_name and not module_name.startswith(exc.name + "."):
            raise
if obj is None:
    raise ModuleNotFoundError({target!r})
for attr in attrs:
    obj = getattr(obj, attr)
print("OBJECT", {target!r})
print("SIGNATURE", inspect.signature(obj))
print("SOURCE", inspect.getsourcefile(obj))
"""
        return f"python -c {shlex.quote(code)}"

    path = Path(target)
    if not target or path.is_absolute() or ".." in path.parts:
        raise ValueError("path probe target must stay inside the workspace")
    if kind == "path_list":
        code = f"""from pathlib import Path
root = Path.cwd()
path = root / {target!r}
print("PATH", path.relative_to(root), "RESOLVED", path.resolve())
print("EXISTS", path.exists(), "FILE", path.is_file(), "DIR", path.is_dir())
if path.is_file():
    print(path.relative_to(root))
elif path.is_dir():
    for index, item in enumerate(sorted(p for p in path.rglob("*") if p.is_file())):
        if index >= 100:
            print("... output capped at 100 files")
            break
        print(item.relative_to(root))
"""
        return f"python -c {shlex.quote(code)}"
    if kind == "cli_help":
        if path.suffix != ".py":
            raise ValueError("cli_help target must be a workspace-relative .py file")
        return f"python {shlex.quote(target)} --help"
    raise ValueError(f"unsupported runtime probe kind: {kind!r}")


def runtime_probe_observation(run: Any, clip) -> str:
    status = (
        f"timed out after {run.duration_s:.0f}s"
        if run.timed_out
        else f"exit {run.exit_code} in {run.duration_s:.0f}s"
    )
    return (
        f"Restricted runtime probe ({status}).\n"
        f"stdout:\n{clip(run.stdout, 4000)}\n"
        f"stderr:\n{clip(run.stderr, 4000)}"
    )


# Backward-compatible private names used by older tests/imports.
_runtime_probe_command = runtime_probe_command
_runtime_probe_observation = runtime_probe_observation
