from __future__ import annotations

import sys
from pathlib import Path

import pytest

from exec.session import Session


def test_session_injects_explicit_non_secret_environment(tmp_path: Path) -> None:
    session = Session(
        tmp_path / "ws",
        venv_python=sys.executable,
        extra_env={"REPRO_DEVICE": "mps", "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )

    run = session.shell(
        "python -c \"import os; print(os.environ['REPRO_DEVICE'])\""
    )

    assert run.ok
    assert run.stdout.strip() == "mps"


def test_session_rejects_explicit_secret_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="secret-like"):
        Session(tmp_path / "ws", extra_env={"LLM_API_KEY": "do-not-inject"})
