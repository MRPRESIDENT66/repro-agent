"""MCP server exposing Repro-Agent's core tools to any MCP client.

The reproduction agent's toolbelt — repo search, restricted runtime probing, and
sandboxed command execution — is exposed here over the Model Context Protocol, so
an MCP client (Claude Desktop, Claude Code, Cursor, ...) can drive the same tools
the in-process agent uses.

Run it:

    python mcp_server.py            # stdio transport (default)

Register it with an MCP client (e.g. Claude Code `.mcp.json`):

    {
      "mcpServers": {
        "repro-agent": {
          "command": "python",
          "args": ["mcp_server.py"],
          "cwd": "/path/to/repro-agent"
        }
      }
    }
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make project imports work no matter what directory an MCP client launches from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("repro-agent")


@mcp.tool()
def search_repo(query: str, repo_path: str, k: int = 5) -> str:
    """Search a code repository for files/snippets relevant to a natural-language
    query, using BM25 lexical search + path/symbol signals + LLM reranking.

    Args:
        query: what to look for, e.g. "where is the evaluation entry point".
        repo_path: absolute path to the repository to search.
        k: max number of files to return.

    Requires an OpenAI-compatible LLM via the usual ``LLM_API_KEY`` /
    ``LLM_BASE_URL`` / ``LLM_MODEL`` environment variables.
    """
    from agent.llm import ChatLLM
    from retrieval.search import search_repo as _search_repo

    return _search_repo(query, repo_path, ChatLLM(), k=k)


@mcp.tool()
def runtime_probe(kind: str, target: str) -> str:
    """Run a restricted runtime probe — never full code execution.

    A probe answers a narrow uncertainty about the environment without running an
    untrusted program: does an import work, what is a function's signature, what
    is in a directory, what does a CLI's ``--help`` say.

    Args:
        kind: one of ``import_smoke``, ``python_signature``, ``path_list``,
            ``cli_help``.
        target: the module/symbol/path/command to probe (e.g. ``json.dumps``).

    Returns the probe's captured output.
    """
    from agent.runtime_probe import runtime_probe_command
    from exec.session import Session

    command = runtime_probe_command(kind, target)
    with tempfile.TemporaryDirectory() as workdir:
        run = Session(workdir, venv_python=sys.executable).shell(command, timeout=30)
    stdout = (run.stdout or "").strip()
    stderr = (run.stderr or "").strip()
    if run.ok:
        return stdout or "(no output)"
    return f"{stdout}\n[stderr] {stderr}".strip()


@mcp.tool()
def run_in_sandbox(command: str, timeout: int = 60) -> str:
    """Run a shell command in an isolated subprocess session (its own temp
    working directory), and return the exit code with captured stdout/stderr.

    Intended for evaluation/probe commands, not arbitrary host access.
    """
    from exec.session import Session

    with tempfile.TemporaryDirectory() as workdir:
        run = Session(workdir, venv_python=sys.executable).shell(command, timeout=timeout)
    return (
        f"exit_code: {run.exit_code}\n"
        f"--- stdout ---\n{(run.stdout or '').strip()}\n"
        f"--- stderr ---\n{(run.stderr or '').strip()}"
    )


if __name__ == "__main__":
    mcp.run()
