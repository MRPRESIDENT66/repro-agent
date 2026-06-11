"""Docker execution backend — the hardened option for UNTRUSTED repos.

Same interface as exec.session.Session (shell / read_file / write_file /
transcript / replay), but each command runs inside a persistent, locked-down
container:

  * **resource caps** — memory (+ no swap), CPU, PID limit.
  * **no privileges** — ``cap_drop=ALL``, ``no-new-privileges``.
  * **two-phase network** — the container starts *connected* so a Provision
    phase can clone/pip/download (verify hashes), then :meth:`go_offline`
    disconnects it so the Execution phase runs the repo's code with **no
    network** (no exfiltration). This is the honest answer to "running an
    arbitrary GitHub repo is running untrusted code" that the subprocess
    backend can't give. (Mac note: Docker here is CPU-only — no MPS.)
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from exec.session import RunResult


class DockerSession:
    def __init__(
        self,
        workdir: str | Path,
        image: str = "python:3.12-slim",
        mem: str = "2g",
        cpus: float = 2.0,
        default_timeout: int = 300,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.name = "repro-" + uuid.uuid4().hex[:8]
        self.default_timeout = default_timeout
        self.offline = False
        self.transcript: list[RunResult] = []
        subprocess.run(
            ["docker", "run", "-d", "--name", self.name,
             "-v", f"{self.workdir}:/workspace", "-w", "/workspace",
             "--memory", mem, "--memory-swap", mem, f"--cpus={cpus}",
             "--pids-limit", "256", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges",
             image, "sleep", "infinity"],
            check=True, capture_output=True, text=True,
        )

    def shell(self, command: str, timeout: int | None = None) -> RunResult:
        timeout = timeout or self.default_timeout
        start = time.monotonic()
        try:
            p = subprocess.run(
                ["docker", "exec", self.name, "bash", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            r = RunResult(command, p.stdout, p.stderr, p.returncode, False, time.monotonic() - start)
        except subprocess.TimeoutExpired as e:
            r = RunResult(command, e.stdout or "", e.stderr or "", -1, True, time.monotonic() - start)
        self.transcript.append(r)
        return r

    def go_offline(self) -> None:
        """End the Provision phase: cut the container's network for Execution."""
        subprocess.run(["docker", "network", "disconnect", "bridge", self.name], capture_output=True)
        self.offline = True

    def read_file(self, path: str) -> str:
        f = self.workdir / path
        return f.read_text(errors="replace") if f.exists() else f"ERROR: not found: {path}"

    def write_file(self, path: str, content: str) -> None:
        f = self.workdir / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")

    def replay_script(self) -> str:
        return "\n".join(r.command for r in self.transcript)

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
