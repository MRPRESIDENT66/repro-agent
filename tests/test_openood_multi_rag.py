from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall
from agent.contracts import (
    extract_python as _extract_python,
    review_requires_repair as _review_requires_repair,
    validate_review as _validate_review,
)
from agent.pipeline import _dynamic_rag_role
from agent.repair import apply_code_patch as _apply_code_patch
from agent.roles import _missing_path_hints
from agent.runtime_probe import runtime_probe_command as _runtime_probe_command
from evals.oracles.openood_ebo import (
    _ID_COUNT,
    _OOD,
    _RUNS,
    _make_public_contract_diagnostics,
    _recompute,
)
from exec.session import Session

def _session(*, ok: bool = True, stderr: str = "") -> SimpleNamespace:
    run = SimpleNamespace(ok=ok, command="python eval_ebo.py", stdout="", stderr=stderr)
    return SimpleNamespace(transcript=[run])


def _write_scores(
    workdir: Path,
    *,
    id_count: int = _ID_COUNT,
    ood_counts: dict[str, int] | None = None,
    ood_score: float = 1.0,
) -> None:
    ood_counts = ood_counts or _OOD
    data = {
        run: {
            "id": [0.0] * id_count,
            **{name: [ood_score] * count for name, count in ood_counts.items()},
        }
        for run in _RUNS
    }
    (workdir / "predictions.json").write_text(json.dumps(data))


def _contract(workdir: Path):
    diagnostics = _make_public_contract_diagnostics(workdir)
    return diagnostics, lambda session: not diagnostics(session)


def _valid_code() -> str:
    return """
import json
from openood.networks import ResNet18_32x32
from torch.utils.data import DataLoader
root_flag = "--root"
json.dump({}, open("predictions.json", "w"))
"""


# ---------------------------------------------------------------------------
# Public verifier contract
# ---------------------------------------------------------------------------

def test_public_contract_rejects_incomplete_id_scores(tmp_path: Path) -> None:
    _write_scores(tmp_path, id_count=2)
    _, passes = _contract(tmp_path)

    assert not passes(_session())
    assert _recompute(tmp_path) is None


def test_public_contract_accepts_complete_score_coverage(tmp_path: Path) -> None:
    _write_scores(tmp_path)
    _, passes = _contract(tmp_path)

    assert passes(_session())
    assert _recompute(tmp_path) == (100.0, 50379)


def test_missing_predictions_prioritizes_latest_execution_error(tmp_path: Path) -> None:
    diagnostics, _ = _contract(tmp_path)
    issues = diagnostics(_session(ok=False, stderr="FileNotFoundError: missing image"))

    assert len(issues) == 1
    assert "No `predictions.json`" in issues[0]
    assert "FileNotFoundError: missing image" in issues[0]


def test_review_status_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "review.md"
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: REPAIR_REQUIRED\n")
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: PASS\n")
    assert not _review_requires_repair(report)


# ---------------------------------------------------------------------------
# Dynamic RAG and probe handling
# ---------------------------------------------------------------------------

def test_dynamic_rag_query_is_generated_from_error_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.pipeline as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    (workspace / "config.yml").write_text("data_root: data/images_classic\n")
    query = "resolve FileNotFoundError benchmark data path"
    report = "Grounded path audit. " + ("x" * 310) + "\nREVIEW_STATUS: REPAIR_REQUIRED"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": query})]),
        Reply("", [ToolCall("s1", "submit_review", {"content": report})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])

    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module,
        "search_repo",
        lambda actual_query, root, llm, **kwargs: (
            "Most relevant files:\n  config.yml  —  dataset configuration"
            if actual_query == query
            else "unexpected query"
        ),
    )

    role, rag = _dynamic_rag_role(
        name="reviewer_test",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=Session(workspace),
        instruction="Query the concrete execution error, then submit the review.",
        context="Execution failed: FileNotFoundError for benchmark data.",
        output_path=workspace / "review.md",
        submit_name="submit_review",
        submit_description="Submit review.",
        validator=_validate_review,
        trigger="execution_error",
        max_steps=3,
    )

    assert rag["dynamic"] is True
    assert rag["queries"] == [query]
    assert role["tool_counts"] == {"search_repo": 1, "submit_review": 1}
    trace = (artifacts / "reviewer_test_rag_trace.md").read_text()
    assert query in trace
    assert "data_root: data/images_classic" in trace


def test_restricted_runtime_probe_is_audited_and_not_an_eval_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.pipeline as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("p1", "runtime_probe", {"kind": "python_signature", "target": "json.dumps"})]),
        Reply("", [ToolCall("q1", "search_repo", {"query": "find official evaluation entry"})]),
        Reply("", [ToolCall("s1", "submit_handoff", {"content": "grounded"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(module, "search_repo", lambda *args, **kwargs: "Most relevant files:\n")
    session = Session(workspace, venv_python=sys.executable)

    role, _ = _dynamic_rag_role(
        name="probe_test",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=session,
        instruction="Query, maybe probe, then submit.",
        context="Execution failed: import uncertainty.",
        output_path=workspace / "handoff.md",
        submit_name="submit_handoff",
        submit_description="Submit handoff.",
        validator=lambda content: content,
        trigger="runtime_uncertainty",
        search_extra_exclude=set(),
        max_steps=4,
        allow_runtime_probe=True,
    )

    assert role["runtime_probes"] == 1
    assert role["runtime_probe_required"] is False
    assert role["runtime_probe_hint"] is None
    trace = (artifacts / "probe_test_probe_trace.md").read_text()
    assert "python_signature `json.dumps`" in trace
    assert "SIGNATURE" in trace


def test_generic_runtime_error_repair_probe_hint_is_soft() -> None:
    command = _runtime_probe_command("python_signature", "json.dumps")
    assert command.startswith("python -c ")
    assert "json.dumps" in command
    assert "import inspect" in command


def test_python_signature_probe_resolves_class_attributes(tmp_path: Path) -> None:
    script = tmp_path / "mod.py"
    script.write_text("class A:\n    x = 1\n")
    assert _extract_python(script.read_text()) == "class A:\n    x = 1\n"


# ---------------------------------------------------------------------------
# Patch-first repair and validation
# ---------------------------------------------------------------------------

def test_code_patch_applies_unique_incremental_replacement(tmp_path: Path) -> None:
    code_path = tmp_path / "eval_ebo.py"
    code_path.write_text(_valid_code() + "\ndata_aux_preprocessor=None\n")
    payload = json.dumps({
        "edits": [{
            "old": "data_aux_preprocessor=None",
            "new": "data_aux_preprocessor=preprocessor",
        }],
        "rationale": "Use the required auxiliary preprocessor.",
    })

    patched = _apply_code_patch(code_path, payload, validate_code=lambda s: s)

    assert "data_aux_preprocessor=preprocessor" in patched
    assert "from openood.networks import ResNet18_32x32" in patched


def test_code_patch_rejects_ambiguous_or_whole_file_replacement(tmp_path: Path) -> None:
    code_path = tmp_path / "eval_ebo.py"
    code_path.write_text(_valid_code() + "\nduplicate = True\nduplicate = True\n")
    ambiguous = json.dumps({
        "edits": [{"old": "duplicate = True", "new": "duplicate = False"}],
        "rationale": "ambiguous",
    })
    with pytest.raises(ValueError, match="exactly once"):
        _apply_code_patch(code_path, ambiguous, validate_code=lambda s: s)

    whole_file = json.dumps({
        "edits": [{"old": code_path.read_text(), "new": _valid_code()}],
        "rationale": "rewrite",
    })
    with pytest.raises(ValueError, match="too much"):
        _apply_code_patch(code_path, whole_file, validate_code=lambda s: s)


def test_code_patch_can_enforce_diagnostic_scope(
    tmp_path: Path,
) -> None:
    code_path = tmp_path / "eval_ebo.py"
    code_path.write_text(_valid_code() + "\ndatasets = {'cifar100': 3}\n")

    unrelated = json.dumps({
        "edits": [{"old": "root_flag = \"--root\"", "new": "root_flag = '--root'"}],
        "rationale": "unrelated",
    })
    with pytest.raises(ValueError, match="does not address"):
        _apply_code_patch(
            code_path,
            unrelated,
            validate_code=lambda s: s,
            required_change_terms={"datasets", "cifar100", "tin"},
        )


def test_missing_path_diagnostic_lists_real_sibling_candidates(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "data" / "benchmark_imglist" / "cifar10"
    parent.mkdir(parents=True)
    for name in ("test_cifar10.txt", "test_cifar100.txt", "test_tin.txt"):
        (parent / name).write_text("x")
    context = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'data/benchmark_imglist/cifar10/test.txt'"
    )

    hints = _missing_path_hints(context, tmp_path)

    assert hints[0].endswith("test_cifar10.txt")
    assert all("test.txt" not in hint for hint in hints)


def test_missing_path_hint_walks_up_to_real_ancestor_on_wrong_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "data" / "images_classic" / "cifar10").mkdir(parents=True)
    (tmp_path / "data" / "benchmark_imglist").mkdir(parents=True)
    context = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/workspace/data/images/cifar10/cifar10/test/airplane/0298.png'"
    )

    hints = _missing_path_hints(context, tmp_path)

    assert any("images_classic" in hint for hint in hints)
    assert all(hint.startswith("data/") for hint in hints)
