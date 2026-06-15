"""DockerSession security properties (skipped when no Docker daemon)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_docker_up = subprocess.run(["docker", "ps"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not _docker_up, reason="Docker daemon not reachable")

from exec.docker_session import DockerSession  # noqa: E402

_NET = (
    "python -c \"import urllib.request as u; "
    "u.urlopen('https://github.com', timeout=8); print('ONLINE')\""
)


def test_two_phase_network(tmp_path: Path) -> None:
    s = DockerSession(tmp_path / "ws")
    try:
        assert "ONLINE" in s.shell(_NET).stdout          # Provision phase: connected
        s.go_offline()
        off = s.shell(_NET)
        assert "ONLINE" not in off.stdout and not off.ok  # Execution phase: cut off
    finally:
        s.close()


def test_state_persists_in_container(tmp_path: Path) -> None:
    s = DockerSession(tmp_path / "ws")
    try:
        s.shell("pip --version > tools.txt 2>&1")
        assert "pip" in s.shell("cat tools.txt").stdout    # second exec sees the first's writes
    finally:
        s.close()


def test_probe_is_not_verifier_visible(tmp_path: Path) -> None:
    s = DockerSession(tmp_path / "ws")
    try:
        assert s.probe("python -c \"print('probe')\"").ok
        assert s.shell("python -c \"print('evaluation')\"").ok
        assert len(s.probe_transcript) == 1
        assert len(s.transcript) == 1
        assert "probe" in s.probe_replay_script()
        assert "evaluation" in s.replay_script()
    finally:
        s.close()


def test_sync_file_waits_for_bind_mount_visibility(tmp_path: Path) -> None:
    s = DockerSession(tmp_path / "ws")
    try:
        s.write_file("generated.py", "print('ok')\n")
        assert s.sync_file("generated.py")
        assert not s.sync_file("missing.py", timeout=0.1)
    finally:
        s.close()
