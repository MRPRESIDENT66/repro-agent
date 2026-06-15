import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall
from agent.multi_rag import (
    _apply_code_patch,
    _dynamic_rag_role,
    _extract_python,
    _missing_path_hints,
    _patch_tool,
    _review_requires_repair,
    _runtime_probe_command,
    _validate_review,
)
from evals.oracles.openood_ebo import (
    NORMALIZATION_SOURCE_REL,
    _ID_COUNT,
    _OOD,
    _RUNS,
    _diagnostic_change_terms,
    _make_public_contract_diagnostics,
    _make_validate_code,
    _normalization_diagnostics_for_code,
    _recompute,
)
from exec.session import Session

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_NO_WD = Path("/nonexistent")  # workdir that skips normalization check (file absent)
_validate_code = _make_validate_code(_NO_WD)
_round_code_is_endorsed = (
    lambda run_ok, contract_passes, review_path:
    run_ok and contract_passes and not _review_requires_repair(review_path)
)
_repair_loop_should_continue = lambda contract_passes: not contract_passes


def _normalization_diagnostics(workdir: Path | None = None) -> list[str]:
    if workdir is None:
        return []
    generated = workdir / "eval_ebo.py"
    if not generated.is_file():
        return []
    return _normalization_diagnostics_for_code(
        generated.read_text(errors="replace"),
        workdir / NORMALIZATION_SOURCE_REL,
    )


def _session(*, ok: bool = True, stderr: str = "") -> SimpleNamespace:
    run = SimpleNamespace(
        ok=ok,
        command="python eval_ebo.py",
        stdout="",
        stderr=stderr,
    )
    return SimpleNamespace(transcript=[run])


def _write_scores(
    workdir: Path,
    *,
    id_count: int = _ID_COUNT,
    ood_counts: dict[str, int] | None = None,
    ood_score: float = 1.0,
) -> None:
    """Write a compact deterministic V2 score artifact for contract tests."""
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


# ---------------------------------------------------------------------------
# Contract tests
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


def test_below_chance_result_flagged_as_inverted_without_leaking_target(
    tmp_path: Path,
) -> None:
    diagnostics, _ = _contract(tmp_path)
    _write_scores(tmp_path, ood_score=-1.0)
    inverted = diagnostics(_session())
    assert any("below" in d.lower() and "chance" in d.lower() for d in inverted)
    assert all("87.58" not in d for d in inverted)

    _write_scores(tmp_path)
    correct = diagnostics(_session())
    assert not any("chance" in d.lower() for d in correct)


def test_public_contract_diagnostics_explain_counts_without_private_target(
    tmp_path: Path,
) -> None:
    _write_scores(tmp_path, ood_counts={"cifar100": 3, "tin": 3})
    diagnostics, _ = _contract(tmp_path)

    issues = diagnostics(_session())

    assert len(issues) == 1
    assert f"id` (exactly {_ID_COUNT} scores)" in issues[0]
    assert "cifar100` (exactly 9000 scores)" in issues[0]
    assert "silently dropped" in issues[0]
    assert "87.58" not in issues[0]


def test_count_mismatch_surfaces_silent_drop_signals_from_log(tmp_path: Path) -> None:
    _write_scores(tmp_path, ood_counts={"cifar100": 9000, "tin": 6526})
    diagnostics, _ = _contract(tmp_path)
    session = _session(stderr=(
        "ERROR:root:[/workspace/data/.../val_7.JPEG] broken\n"
        "FileNotFoundError: [Errno 2] No such file: val_9.JPEG\n"
    ))

    issues = diagnostics(session)

    assert any("drop/error signal" in d for d in issues)
    assert any("silently dropped" in d for d in issues)


def test_missing_predictions_prioritizes_latest_execution_error(tmp_path: Path) -> None:
    diagnostics, _ = _contract(tmp_path)
    issues = diagnostics(_session(ok=False, stderr="FileNotFoundError: missing image"))

    assert len(issues) == 1
    assert "No `predictions.json`" in issues[0]
    assert "FileNotFoundError: missing image" in issues[0]


def test_normalization_diagnostics_compare_generated_code_to_repo(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    (tmp_path / "eval_ebo.py").write_text(
        "from torchvision import transforms\n"
        "cifar10_mean = (0.4914, 0.4822, 0.4465)\n"
        "cifar10_std = (0.2023, 0.1994, 0.201)\n"
        "normalize = transforms.Normalize(cifar10_mean, cifar10_std)\n"
    )
    diagnostics = _normalization_diagnostics(tmp_path)

    assert len(diagnostics) == 1
    assert "normalization mismatch" in diagnostics[0]
    assert "0.247" in diagnostics[0]


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


# ---------------------------------------------------------------------------
# Endorsement / repair-loop gate tests
# ---------------------------------------------------------------------------

def test_round_code_endorsement_requires_all_three_signals(tmp_path: Path) -> None:
    review = tmp_path / "review_report.md"
    passing = tmp_path / "pass.md"
    passing.write_text("Matches repository semantics.\nREVIEW_STATUS: PASS\n")

    review.write_text("The EBO energy sign looks inverted.\nREVIEW_STATUS: REPAIR_REQUIRED\n")
    assert not _round_code_is_endorsed(True, True, review)
    assert not _round_code_is_endorsed(True, False, passing)
    assert not _round_code_is_endorsed(False, True, passing)
    assert _round_code_is_endorsed(True, True, passing)
    assert not _round_code_is_endorsed(True, True, tmp_path / "missing.md")


def test_repair_loop_stops_once_contract_passes_regardless_of_reviewer(
    tmp_path: Path,
) -> None:
    _, passes = _contract(tmp_path)
    _write_scores(tmp_path)
    assert not _repair_loop_should_continue(passes(_session()))

    _write_scores(tmp_path, id_count=2)
    assert _repair_loop_should_continue(passes(_session()))


def test_review_status_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "review.md"
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: REPAIR_REQUIRED\n")
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: PASS\n")
    assert not _review_requires_repair(report)


# ---------------------------------------------------------------------------
# Dynamic RAG role tests
# ---------------------------------------------------------------------------

def test_dynamic_rag_query_is_generated_from_error_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.multi_rag as module

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
    rag_llm = ScriptedLLM([])
    llms = iter([role_llm, rag_llm, ScriptedLLM([])])

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
    assert rag["trigger"] == "execution_error"
    assert role["tool_counts"] == {"search_repo": 1, "submit_review": 1}
    trace = (artifacts / "reviewer_test_rag_trace.md").read_text()
    assert query in trace
    assert "data_root: data/images_classic" in trace


def test_dynamic_rag_forces_synthesis_after_query_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    report = "Grounded handoff. " + ("x" * 310)
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "find official evaluation entry"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([Reply(report)])])

    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module,
        "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )

    role, rag = _dynamic_rag_role(
        name="navigator_test",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=Session(workspace),
        instruction="Search, then synthesize.",
        context="Find the official evaluation path.",
        output_path=workspace / "handoff.md",
        submit_name="submit_handoff",
        submit_description="Submit handoff.",
        validator=lambda content: content,
        trigger="initial_task",
        max_steps=3,
        max_queries=1,
    )

    assert (workspace / "handoff.md").read_text() == report
    assert rag["queries"] == ["find official evaluation entry"]
    assert role["steps"] == 2


def test_restricted_runtime_probe_is_audited_and_not_an_eval_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("p1", "runtime_probe", {
            "kind": "python_signature",
            "target": "json.dumps",
        })]),
        Reply("", [ToolCall("q1", "search_repo", {
            "query": "find official evaluation entry",
        })]),
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
        instruction="Investigate then submit.",
        context="Find the runtime interface.",
        output_path=workspace / "handoff.md",
        submit_name="submit_handoff",
        submit_description="Submit handoff.",
        validator=lambda content: content,
        trigger="initial_task",
        max_steps=4,
        allow_runtime_probe=True,
    )

    assert session.transcript == []
    assert len(session.probe_transcript) == 1
    assert "SIGNATURE" in session.probe_transcript[0].stdout
    assert role["runtime_probes"] == 1
    assert role["tool_counts"] == {
        "runtime_probe": 1,
        "search_repo": 1,
        "submit_handoff": 1,
    }
    assert (artifacts / "probe_test_probe_trace.md").exists()


def test_generic_runtime_error_repair_requires_probe_before_submit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {
            "query": "inspect failing optional import",
        })]),
        Reply("", [ToolCall("early", "submit_code", {"content": "fixed = True\n"})]),
        Reply("", [ToolCall("p1", "runtime_probe", {
            "kind": "import_smoke",
            "target": "json",
        })]),
        Reply("", [ToolCall("s1", "submit_code", {"content": "fixed = True\n"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(module, "search_repo", lambda *args, **kwargs: "Most relevant files:\n")
    session = Session(workspace, venv_python=sys.executable)

    role, _ = _dynamic_rag_role(
        name="repair_1",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=session,
        instruction="Repair from evidence.",
        context="ModuleNotFoundError: No module named 'optional_dep'",
        output_path=workspace / "eval.py",
        submit_name="submit_code",
        submit_description="Submit code.",
        validator=lambda content: content,
        trigger="execution_error_and_reviewer_finding",
        max_steps=5,
        allow_runtime_probe=True,
    )

    assert role["runtime_probe_required"] is True
    assert role["runtime_probes"] == 1
    assert role["format_errors"] == 1
    assert (workspace / "eval.py").read_text() == "fixed = True\n"


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        ("import_smoke", "os;system"),
        ("python_signature", "json.dumps()"),
        ("path_list", "../private"),
        ("path_list", "/etc"),
        ("cli_help", "README.md"),
        ("shell", "ls"),
    ],
)
def test_restricted_runtime_probe_rejects_raw_or_escaping_targets(kind, target) -> None:
    with pytest.raises(ValueError):
        _runtime_probe_command(kind, target)


def test_python_signature_probe_resolves_class_attributes(tmp_path: Path) -> None:
    session = Session(tmp_path, venv_python=sys.executable)

    run = session.probe(
        _runtime_probe_command("python_signature", "json.JSONEncoder.__init__"),
        timeout=10,
    )

    assert run.ok
    assert "SIGNATURE" in run.stdout
    assert "json.JSONEncoder.__init__" in run.stdout


def test_path_list_probe_allows_provisioned_workspace_symlink(tmp_path: Path) -> None:
    external = tmp_path / "provisioned"
    external.mkdir()
    (external / "asset.bin").write_text("data")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "assets").symlink_to(external, target_is_directory=True)
    session = Session(workspace, venv_python=sys.executable)

    run = session.probe(_runtime_probe_command("path_list", "assets"), timeout=10)

    assert run.ok
    assert "assets/asset.bin" in run.stdout


def test_openood_image_pins_faiss_for_optional_import_chain() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1] / "docker" / "openood.Dockerfile"
    ).read_text()

    assert "numpy==1.26.4" in dockerfile
    assert "faiss-cpu==1.7.4" in dockerfile
    assert "import torch, torchvision, numpy, sklearn, pandas, faiss" in dockerfile


def _synthesis_harness(tmp_path, monkeypatch, synthesis_replies, validator):
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "find the official evaluation entry"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM(synthesis_replies)])
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module, "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )
    with pytest.raises(RuntimeError, match="failed to synthesize"):
        _dynamic_rag_role(
            name="synth",
            task="Test task",
            workdir=workspace,
            artifact_dir=artifacts,
            session=Session(workspace),
            instruction="Search, then synthesize.",
            context="Find the official evaluation path.",
            output_path=workspace / "artifact.txt",
            submit_name="submit_handoff",
            submit_description="Submit handoff.",
            validator=validator,
            trigger="initial_task",
            max_steps=3,
            max_queries=1,
            synthesis_attempts=len(synthesis_replies),
        )
    return (artifacts / "synth_synthesis_transcript.jsonl").read_text()


def test_synthesis_rejects_no_progress_resubmission(tmp_path: Path, monkeypatch) -> None:
    def always_fail(content: str) -> str:
        raise ValueError("artifact must define an evaluation entry")

    identical = "x" * 400
    transcript = _synthesis_harness(
        tmp_path, monkeypatch, [Reply(identical)] * 3, always_fail
    )
    assert "barely changed" in transcript


def test_synthesis_accepts_valid_near_identical_fix(tmp_path: Path, monkeypatch) -> None:
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    bad = "x" * 400 + "\nstd = [0.2023, 0.1994, 0.2010]\n"
    good = "x" * 400 + "\nstd = [0.247, 0.2435, 0.2616]\n"

    def validator(content: str) -> str:
        if "0.247" not in content:
            raise ValueError("normalization mismatch")
        return content

    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "repository normalization constants"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([Reply(bad), Reply(good)])])
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(module, "search_repo", lambda q, r, l, **k: "Most relevant files:\n")

    _dynamic_rag_role(
        name="synth_fix",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=Session(workspace),
        instruction="Search then synthesize.", context="fix normalization",
        output_path=workspace / "artifact.txt", submit_name="submit_handoff",
        submit_description="submit", validator=validator, trigger="initial_task",
        max_steps=3, max_queries=1, synthesis_attempts=2,
    )

    assert "0.247" in (workspace / "artifact.txt").read_text()


def test_synthesis_escalates_on_repeated_validation_error(tmp_path: Path, monkeypatch) -> None:
    def always_fail(content: str) -> str:
        raise ValueError("artifact must define an evaluation entry")

    transcript = _synthesis_harness(
        tmp_path, monkeypatch, [Reply("a" * 400), Reply("b" * 400)], always_fail
    )
    assert "SAME error" in transcript


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

def test_review_validator_normalizes_markdown_status() -> None:
    review = "Grounded audit. " + ("x" * 310) + "\n\n**REVIEW_STATUS: REPAIR_REQUIRED**"
    normalized = _validate_review(review)
    assert normalized.endswith("REVIEW_STATUS: REPAIR_REQUIRED\n")


def test_code_validator_rejects_model_reimplementation_and_second_root_flag() -> None:
    code = """
import json
from openood.networks import ResNet18_32x32
from torch.utils.data import DataLoader
class ResNet18_32x32: pass
checkpoint = "--root"
other = "--checkpoint_root"
paths = "data/benchmark_imglist data/images_classic"
runs = ["s0", "s1", "s2"]
datasets = ["cifar100", "tin"]
score = logsumexp(x)
json.dump({}, open("predictions.json", "w"))
"""
    with pytest.raises(ValueError, match="fixed model/CLI contract"):
        _validate_code(code)


def test_extract_python_picks_the_real_script_not_a_preface_snippet() -> None:
    reply = (
        "Here is the EBO idea:\n\n"
        "```python\nscore = logsumexp(logits)  # illustrative snippet\n```\n\n"
        "Now the full evaluation script:\n\n"
        "```python\nimport json\n# ... full eval ...\n"
        "json.dump(scores, open('predictions.json', 'w'))\n```\n"
    )
    code = _extract_python(reply)
    assert "predictions.json" in code and "json.dump" in code
    assert "illustrative snippet" not in code


def test_code_validator_rejects_syntax_error_before_execution() -> None:
    with pytest.raises(ValueError, match="syntactically valid"):
        _validate_code(_valid_code() + "\nif True print('broken')\n")


def test_code_validator_allows_forbidden_names_in_comments_and_strings() -> None:
    code = (
        '"""This module deliberately avoids openood.evaluation_api and never '
        'parses checkpoint config.yml files."""\n'
        + _valid_code()
        + "\n# It replicates TestStandardPreProcessor logic without importing"
        + " that class.\n"
        + 'note = "do not pass --checkpoint_root; use --root instead"\n'
        + 'doc = "ImglistDataset is reused from the repository, not redefined"\n'
    )
    assert "TestStandardPreProcessor" in code
    _validate_code(code)


def test_code_validator_rejects_real_forbidden_usage_at_ast_level() -> None:
    with pytest.raises(ValueError, match="forbidden import of 'TestStandardPreProcessor'"):
        _validate_code(
            _valid_code()
            + "\nfrom openood.preprocessors import TestStandardPreProcessor\n"
        )
    with pytest.raises(ValueError, match="forbidden instantiation 'TestStandardPreProcessor"):
        _validate_code(_valid_code() + "\npre = TestStandardPreProcessor(cfg)\n")
    with pytest.raises(ValueError, match="forbidden re-implementation 'class ImglistDataset'"):
        _validate_code(_valid_code() + "\nclass ImglistDataset:\n    pass\n")
    with pytest.raises(ValueError, match="forbidden call argument 'config.yml'"):
        _validate_code(_valid_code() + "\ncfg = load_config(open('s0/config.yml'))\n")
    with pytest.raises(ValueError, match="forbidden call argument '--checkpoint_root'"):
        _validate_code(_valid_code() + "\nparser.add_argument('--checkpoint_root')\n")


def test_normalization_diagnostics_resolve_function_local_variables(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    validate = _make_validate_code(tmp_path)
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "def build():\n"
        + "    mean = [0.4914, 0.4822, 0.4465]\n"
        + "    std = [0.2023, 0.1994, 0.201]\n"
        + "    return transforms.Normalize(mean, std)\n"
    )
    with pytest.raises(ValueError, match="normalization mismatch"):
        validate(code)


def test_normalization_diagnostics_allow_repo_dict_subscript(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    validate = _make_validate_code(tmp_path)
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "from openood.preprocessors.transform import normalization_dict\n"
        + "mean, std = normalization_dict['cifar10']\n"
        + "norm = transforms.Normalize(mean, std)\n"
    )
    assert validate(code)


def test_recompute_rejects_missing_run_block(tmp_path: Path) -> None:
    _write_scores(tmp_path)
    data = json.loads((tmp_path / "predictions.json").read_text())
    data.pop("s2")
    (tmp_path / "predictions.json").write_text(json.dumps(data))

    assert _recompute(tmp_path) is None


def test_code_validator_rejects_broad_optional_dependency_imports() -> None:
    with pytest.raises(ValueError, match="fixed model/CLI contract"):
        _validate_code(
            _valid_code()
            + "\nfrom openood.evaluation_api import Evaluator\n"
        )
    with pytest.raises(ValueError, match="fixed model/CLI contract"):
        _validate_code(
            _valid_code()
            + "\nfrom openood.evaluators.metrics import compute_all_metrics\n"
        )
    with pytest.raises(ValueError, match="fixed model/CLI contract"):
        _validate_code(
            _valid_code()
            + "\nfrom openood.utils.config import Config\n"
            + "preprocessor = TestStandardPreProcessor(Config('config.yml'))\n"
        )


def test_code_validator_rejects_normalization_mismatch(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    validate = _make_validate_code(tmp_path)
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "transform = transforms.Normalize("
        + "mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])\n"
    )
    with pytest.raises(ValueError, match="normalization mismatch"):
        validate(code)


def _valid_code() -> str:
    return """
import json
from openood.networks import ResNet18_32x32
from torch.utils.data import DataLoader
root_flag = "--root"
json.dump({}, open("predictions.json", "w"))
"""


# ---------------------------------------------------------------------------
# Patch tests
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


def test_code_patch_shows_closest_real_code_when_old_text_missing(tmp_path: Path) -> None:
    code_path = tmp_path / "eval_ebo.py"
    code_path.write_text(
        _valid_code()
        + "\nnum_samples = len(loader.dataset)\n"
        + "datasets = {'cifar100': num_samples}\n"
    )
    stale = json.dumps({
        "edits": [{"old": "datasets = {'cifar100': n_samples}", "new": "datasets = {'cifar100': count}"}],
        "rationale": "fix the dataset count source",
    })
    with pytest.raises(ValueError) as exc:
        _apply_code_patch(code_path, stale, validate_code=lambda s: s)
    message = str(exc.value)
    assert "was not found" in message
    assert "Closest actual code" in message
    assert "datasets = {'cifar100': num_samples}" in message


def test_code_patch_protects_confirmed_block_and_enforces_diagnostic_scope(
    tmp_path: Path,
) -> None:
    code_path = tmp_path / "eval_ebo.py"
    confirmed = "data_root = '/workspace/data'"
    code_path.write_text(_valid_code() + f"\n{confirmed}\ndatasets = {{'cifar100': 3}}\n")
    removes_confirmed = json.dumps({
        "edits": [{"old": confirmed, "new": "data_root = '/data'"}],
        "rationale": "regression",
    })
    with pytest.raises(ValueError, match="already confirmed"):
        _apply_code_patch(
            code_path,
            removes_confirmed,
            validate_code=lambda s: s,
            protected_blocks={confirmed},
        )

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

    hides_unrelated_change_in_large_dataset_block = json.dumps({
        "edits": [{
            "old": "datasets = {'cifar100': 3}",
            "new": "datasets = {'cifar100': 3}\nsys.exit(0)",
        }],
        "rationale": "pretends to fix counts",
    })
    with pytest.raises(ValueError, match="does not address"):
        _apply_code_patch(
            code_path,
            hides_unrelated_change_in_large_dataset_block,
            validate_code=lambda s: s,
            required_change_terms={"datasets", "cifar100", "tin", "len("},
        )


def test_dynamic_repair_role_submits_structured_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import agent.multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    code_path = workspace / "eval_ebo.py"
    code_path.write_text(_valid_code() + "\nbroken = True\n")
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "fix broken flag"})]),
        Reply("", [ToolCall("p1", "submit_patch", {
            "edits": [{"old": "broken = True", "new": "broken = False"}],
            "rationale": "Correct the concrete failure.",
        })]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([])])
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module,
        "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )

    role, _ = _dynamic_rag_role(
        name="repair_test",
        task="Test task",
        workdir=workspace,
        artifact_dir=artifacts,
        session=Session(workspace),
        instruction="Search and patch.",
        context="Execution failed because broken is true.",
        output_path=code_path,
        submit_name="submit_patch",
        submit_description="Submit patch.",
        validator=lambda payload: _apply_code_patch(
            code_path, payload, validate_code=lambda s: s
        ),
        trigger="execution_error_and_reviewer_finding",
        submit_schema=_patch_tool("submit_patch", "Submit patch."),
        submission_adapter=lambda arguments: json.dumps(arguments),
    )

    assert "broken = False" in code_path.read_text()
    assert role["tool_counts"] == {"search_repo": 1, "submit_patch": 1}
    assert role["submission_trace"] == "repair_test_submission.json"
    saved = json.loads((artifacts / "repair_test_submission.json").read_text())
    assert saved["edits"][0]["old"] == "broken = True"
