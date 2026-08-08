from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall
from agent.contracts import (
    extract_python as _extract_python,
    validate_audit as _validate_audit,
)
from agent.repair import apply_code_patch as _apply_code_patch
from agent.roles import RoleDeps, missing_path_hints, run_rag_role
from agent.runtime_probe import runtime_probe_command as _runtime_probe_command
from evals.catalog import make_config as make_task_config, manifest_path
from evals.manifest import load_manifest
from exec.session import Session

MANIFEST = load_manifest(manifest_path("openood_ebo"))
GROUPED = MANIFEST.grouped_scores
assert GROUPED is not None
RUNS = GROUPED.groups
ID_COUNT = GROUPED.series[GROUPED.negative_series]
OOD_COUNTS = {
    name: GROUPED.series[name]
    for name in GROUPED.positive_series
}


def _config():
    return make_task_config("openood_ebo", "contract-test")


def _write_scores(
    workdir: Path,
    *,
    id_count: int = ID_COUNT,
    ood_counts: dict[str, int] | None = None,
    ood_score: float = 1.0,
) -> None:
    ood_counts = ood_counts or OOD_COUNTS
    data = {
        run: {
            "id": [0.0] * id_count,
            **{name: [ood_score] * count for name, count in ood_counts.items()},
        }
        for run in RUNS
    }
    (workdir / "predictions.json").write_text(json.dumps(data))


def _valid_code() -> str:
    return """
import json
from openood.networks import ResNet18_32x32
from torch.utils.data import DataLoader
root_flag = "--root"
json.dump({}, open("predictions.json", "w"))
"""


def test_openood_mps_config_is_explicit_and_not_docker_offline(monkeypatch) -> None:
    monkeypatch.setenv("OPENOOD_EXECUTION_BACKEND", "mps")

    config = make_task_config("openood_ebo", "mps_config_test")

    assert config.execution_backend == "mps"
    assert config.session_go_offline is False
    assert "MPS acceleration" in config.task
    assert config.public_execution_command.startswith("REPRO_DEVICE=mps ")


def test_openood_rejects_unknown_execution_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENOOD_EXECUTION_BACKEND", "cuda-ish")

    with pytest.raises(ValueError, match="docker.*mps"):
        make_task_config("openood_ebo", "bad_backend_test")


# ---------------------------------------------------------------------------
# Public verifier contract
# ---------------------------------------------------------------------------

def test_public_contract_rejects_incomplete_id_scores(tmp_path: Path) -> None:
    _write_scores(tmp_path, id_count=2)

    assert _config().recompute_fn(tmp_path) is None


def test_public_contract_accepts_complete_score_coverage(tmp_path: Path) -> None:
    _write_scores(tmp_path)

    config = _config()
    assert config.recompute_fn(tmp_path) == (100.0, 50379)
    assert config.public_check_fn(tmp_path) is True


def test_public_contract_rejects_inverted_score_direction(tmp_path: Path) -> None:
    _write_scores(tmp_path, ood_score=-1.0)

    config = _config()
    issues = config.public_diagnostics_fn(tmp_path)

    assert config.public_check_fn(tmp_path) is False
    assert len(issues) == 6
    assert all("requires OOD scores HIGHER than ID" in issue for issue in issues)
    assert all("87.58" not in issue for issue in issues)


def test_pass_review_requires_source_evidence_for_semantic_pipeline() -> None:
    unsupported = "Plausible review without citations. " + ("x" * 310)
    with pytest.raises(ValueError, match="preprocessing"):
        _validate_audit(unsupported + "\nAUDIT_STATUS: PASS")

    grounded = """Source-grounded audit of the complete evaluation path.
- `model`: `repo/model.py:12` defines the constructor and checkpoint load.
- `data`: `repo/data.py:30` defines the requested test split.
- `preprocessing`: `repo/preprocess.py:8` defines ordered transforms and constants.
- `metric`: `repo/metric.py:20` defines score direction and aggregation.
The execution log confirms that the measured artifact follows those definitions.
AUDIT_STATUS: PASS
"""

    assert _validate_audit(grounded).endswith("AUDIT_STATUS: PASS\n")


def test_pass_review_accepts_documentation_evidence_and_markdown_labels() -> None:
    grounded = """Source-grounded audit of the complete evaluation path.
- Model: loaded with the requested runtime identifier.
- Data: complete validation split loaded without shuffling.
- `model:` `clip/clip.py:94` loads the requested checkpoint.
- **data**: `data/prompts.md:523` defines the classes and prompt templates.
- preprocessing: `clip/clip.py:79-86` defines the ordered image transform.
- `metric`: `eval_clip.py:72` defines cosine similarity and top-1 argmax.
The execution log confirms 10000 measured per-sample predictions.
AUDIT_STATUS: PASS
"""

    assert _validate_audit(grounded).endswith("AUDIT_STATUS: PASS\n")


# ---------------------------------------------------------------------------
# Dynamic RAG and probe handling
# ---------------------------------------------------------------------------

def test_dynamic_rag_query_is_generated_from_error_context(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    (workspace / "config.yml").write_text("data_root: data/images_classic\n")
    query = "resolve FileNotFoundError benchmark data path"
    report = "Grounded path audit. " + ("x" * 310) + "\nAUDIT_STATUS: REPAIR_REQUIRED"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": query})]),
        Reply("", [ToolCall("s1", "submit_audit", {"content": report})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])
    deps = RoleDeps(
        llm_factory=lambda: next(llms),
        search_fn=lambda actual_query, root, llm, **kwargs: (
            "Most relevant files:\n  config.yml  —  dataset configuration"
            if actual_query == query
            else "unexpected query"
        ),
    )

    role, rag = run_rag_role(
        name="auditor_test",
        workdir=workspace,
        artifact_dir=artifacts,
        session=Session(workspace),
        instruction="Query the concrete execution error, then submit the review.",
        context="Execution failed: FileNotFoundError for benchmark data.",
        output_path=workspace / "review.md",
        submit_name="submit_audit",
        submit_description="Submit review.",
        validator=_validate_audit,
        trigger="execution_error",
        max_steps=3,
        deps=deps,
    )

    assert rag["dynamic"] is True
    assert rag["queries"] == [query]
    assert role["tool_counts"] == {"search_repo": 1, "submit_audit": 1}
    trace = (artifacts / "auditor_test_rag_trace.md").read_text()
    assert query in trace
    assert "data_root: data/images_classic" in trace


def test_restricted_runtime_probe_is_audited_and_not_an_eval_command(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("p1", "runtime_probe", {"kind": "python_signature", "target": "json.dumps"})]),
        Reply("", [ToolCall("q1", "search_repo", {"query": "find official evaluation entry"})]),
        Reply("", [ToolCall("s1", "submit_handoff", {"content": "grounded"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])
    deps = RoleDeps(
        llm_factory=lambda: next(llms),
        search_fn=lambda *args, **kwargs: "Most relevant files:\n",
    )
    session = Session(workspace, venv_python=sys.executable)

    role, _ = run_rag_role(
        name="probe_test",
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
        deps=deps,
    )

    assert role["runtime_probes"] == 1
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


def test_python_source_probe_reads_bounded_installed_definition(tmp_path: Path) -> None:
    command = _runtime_probe_command("python_source", "json.dumps")
    command = command.replace("python -c", f"{sys.executable} -c", 1)
    session = Session(tmp_path)
    run = session.probe(command, timeout=10)
    session.close()

    assert run.ok
    assert "OBJECT json.dumps" in run.stdout
    assert "DEFINITION_LINES" in run.stdout
    assert "def dumps" in run.stdout


def test_python_source_probe_lists_class_methods(tmp_path: Path) -> None:
    command = _runtime_probe_command("python_source", "json.JSONEncoder")
    command = command.replace("python -c", f"{sys.executable} -c", 1)
    session = Session(tmp_path)
    run = session.probe(command, timeout=10)
    session.close()

    assert run.ok
    assert "METHODS" in run.stdout
    assert "encode" in run.stdout


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

    hints = missing_path_hints(context, tmp_path)

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

    hints = missing_path_hints(context, tmp_path)

    assert any("images_classic" in hint for hint in hints)
    assert all(hint.startswith("data/") for hint in hints)
