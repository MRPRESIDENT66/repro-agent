"""The MCP server exposes the agent toolbelt and the tools actually run."""

from __future__ import annotations

import asyncio

import mcp_server as srv


def test_tools_are_registered() -> None:
    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert names == {"search_repo", "runtime_probe", "run_in_sandbox"}


def test_runtime_probe_runs_without_api_key() -> None:
    out = srv.runtime_probe("python_signature", "json.dumps")
    assert "SIGNATURE" in out


def test_run_in_sandbox_captures_output() -> None:
    out = srv.run_in_sandbox('python -c "print(6*7)"')
    assert "exit_code: 0" in out
    assert "42" in out
