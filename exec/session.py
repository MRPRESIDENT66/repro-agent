"""Persistent execution session — the lean foundation.

A session is a working directory + a Python env on PATH. Every action runs as a
subprocess with a **clean, secret-scrubbed environment** but **shared on-disk
state**: a `pip install` or a written script in one step persists for the next
(state lives in the filesystem, not the process). That's what makes it a
*session* rather than a string of disposable containers.

This is the M1 backend (subprocess + MPS, for human-reviewed repos). Untrusted
repos get a Docker/VM backend later — same interface.

Every command is recorded so the whole run is replayable and auditable.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Session:
    def __init__(
        self,
        workdir: str | Path,
        venv_python: str | Path | None = None,
        default_timeout: int = 180,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout
        self.transcript: list[RunResult] = []
        self.probe_transcript: list[RunResult] = []
        self._env = self._clean_env(venv_python)

    @staticmethod
    def _clean_env(venv_python: str | Path | None) -> dict[str, str]:
        # Deliberately minimal: no inherited secrets (API keys/tokens), just
        # enough to run Python + reach the user caches (torch.hub / HF datasets).
        path = "/usr/bin:/bin:/usr/sbin:/sbin"
        if venv_python:
            # .absolute() not .resolve(): venv `python` is a symlink to the base
            # interpreter, and resolving it would put the WRONG bin dir on PATH.
            path = f"{Path(venv_python).expanduser().absolute().parent}:{path}"
        env = {"PATH": path, "HOME": os.environ.get("HOME", ""), "LANG": "en_US.UTF-8"}
        # Keep proxy vars if present (network), but never *_KEY/_TOKEN/_SECRET.
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
            if k in os.environ:
                env[k] = os.environ[k]
        return env

    def _run(
        self,
        command: str,
        timeout: int | None,
        transcript: list[RunResult],
    ) -> RunResult:
        timeout = timeout or self.default_timeout
        start = time.monotonic()
        try:
            p = subprocess.run(
                ["bash", "-c", command],
                cwd=self.workdir,
                env=self._env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            r = RunResult(
                command, p.stdout, p.stderr, p.returncode, False, time.monotonic() - start
            )
        except subprocess.TimeoutExpired as e:
            r = RunResult(
                command,
                (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or ""),
                -1,
                True,
                time.monotonic() - start,
            )
        transcript.append(r)
        return r

    def shell(self, command: str, timeout: int | None = None) -> RunResult:
        """Run and record a verifier-visible evaluation command."""
        return self._run(command, timeout, self.transcript)

    def probe(self, command: str, timeout: int | None = None) -> RunResult:
        """Run an audited diagnostic command outside the evaluation transcript."""
        return self._run(command, timeout, self.probe_transcript)

    def read_file(self, path: str) -> str:
        f = self.workdir / path
        if not f.exists():
            return f"ERROR: file not found: {path}"
        return f.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        f = self.workdir / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")

    def sync_file(self, path: str, timeout: float = 5.0) -> bool:
        """Confirm a generated workspace file is visible before execution."""
        del timeout
        return (self.workdir / path).is_file()

    def replay_script(self) -> str:
        """The exact shell commands run, in order — the auditable, replayable log."""
        return "\n".join(r.command for r in self.transcript)

    def probe_replay_script(self) -> str:
        return "\n".join(r.command for r in self.probe_transcript)
