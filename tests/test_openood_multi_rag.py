import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall
from exec.session import Session
from run_openood_multi_rag import (
    _apply_code_patch,
    _diagnostic_change_terms,
    _dynamic_rag_role,
    _missing_path_hints,
    _normalization_diagnostics,
    _patch_tool,
    _public_contract_diagnostics,
    _public_contract_passes,
    _review_requires_repair,
    _round_code_is_endorsed,
    _validate_code,
    _validate_review,
)


def _session(actual: float, run_value: float) -> SimpleNamespace:
    result = {
        "metric": "near_ood_auroc",
        "actual": actual,
        "datasets": {"cifar100": 9000, "tin": 7793},
        "run_metrics": {
            run: {"cifar100": run_value, "tin": run_value}
            for run in ("s0", "s1", "s2")
        },
        "aggregation": "dataset_mean_then_run_mean",
    }
    run = SimpleNamespace(
        ok=True,
        command="python eval_ebo.py",
        stdout=f"REPRO_RESULT {json.dumps(result)}\n",
    )
    return SimpleNamespace(transcript=[run])


def test_public_contract_rejects_fractional_auroc() -> None:
    assert not _public_contract_passes(_session(0.87, 0.87))


def test_public_contract_accepts_percentage_auroc() -> None:
    assert _public_contract_passes(_session(87.0, 87.0))


def test_public_contract_accepts_two_decimal_aggregation_rounding() -> None:
    assert _public_contract_passes(_session(87.09, 87.0916666667))


def test_below_chance_result_flagged_as_inverted_without_leaking_target() -> None:
    # Regression for attempt 029: blind, an inverted detector (12.42 = 100-87.58)
    # is symmetric with the correct value, so the agent oscillated. A general
    # below-chance sanity signal breaks the symmetry — without revealing 87.58.
    inverted = _public_contract_diagnostics(_session(12.42, 12.42))
    assert any("below" in d.lower() and "chance" in d.lower() for d in inverted)
    assert all("87.58" not in d for d in inverted)  # no private target leaked

    # A correct, above-chance result carries no inversion flag (and otherwise
    # passes the contract).
    correct = _public_contract_diagnostics(_session(87.58, 87.58))
    assert not any("chance" in d.lower() for d in correct)


def test_public_contract_diagnostics_explain_counts_without_private_target() -> None:
    session = _session(87.0, 87.0)
    payload = json.loads(session.transcript[0].stdout.removeprefix("REPRO_RESULT "))
    payload["datasets"] = {"cifar100": 3, "tin": 3}
    session.transcript[0].stdout = f"REPRO_RESULT {json.dumps(payload)}\n"

    diagnostics = _public_contract_diagnostics(session)

    assert len(diagnostics) == 1
    assert diagnostics[0].startswith(
        "Dataset counts mismatch: expected {'cifar100': 9000, 'tin': 7793}, "
        "got {'cifar100': 3, 'tin': 3}."
    )
    assert "silently dropped" in diagnostics[0]   # generic count-short guidance
    assert "87.58" not in diagnostics[0]
    assert _diagnostic_change_terms(diagnostics) == {"datasets", "len("}


def test_count_mismatch_surfaces_silent_drop_signals_from_log() -> None:
    # Regression for 027/028/031: a short count because the loader silently
    # skipped unreadable items should be diagnosed from the log, not left a
    # mystery — generically (no dataset/path names).
    session = _session(87.0, 87.0)
    payload = json.loads(session.transcript[0].stdout.removeprefix("REPRO_RESULT "))
    payload["datasets"] = {"cifar100": 9000, "tin": 6526}
    session.transcript[0].stdout = f"REPRO_RESULT {json.dumps(payload)}\n"
    session.transcript[0].stderr = (
        "ERROR:root:[/workspace/data/.../val_7.JPEG] broken\n"
        "FileNotFoundError: [Errno 2] No such file: val_9.JPEG\n"
    )

    diagnostics = _public_contract_diagnostics(session)

    assert any("drop/error signal" in d for d in diagnostics)  # log-derived drop count
    assert any("silently dropped" in d for d in diagnostics)    # generic guidance


def test_public_contract_diagnostics_prioritize_malformed_successful_result() -> None:
    session = SimpleNamespace(
        transcript=[
            SimpleNamespace(
                ok=False,
                command="python eval_ebo.py",
                stdout="",
                stderr="FileNotFoundError: old failure",
            ),
            SimpleNamespace(
                ok=True,
                command="python eval_ebo.py",
                stdout="REPRO_RESULT {'metric': 'near_ood_auroc'}\n",
                stderr="",
            ),
        ],
        workdir=None,
    )

    diagnostics = _public_contract_diagnostics(session)

    assert diagnostics == [
        "A successful evaluation printed REPRO_RESULT, but it was not valid strict "
        "JSON. Serialize the result object with json.dumps."
    ]
    assert _diagnostic_change_terms(diagnostics) == {"json.dumps", "repro_result"}


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
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    parent = tmp_path / "data" / "benchmark_imglist" / "cifar10"
    parent.mkdir(parents=True)
    for name in ("test_cifar10.txt", "test_cifar100.txt", "test_tin.txt"):
        (parent / name).write_text("x")
    monkeypatch.setattr(module, "WORKDIR", tmp_path)
    context = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'data/benchmark_imglist/cifar10/test.txt'"
    )

    hints = _missing_path_hints(context)

    assert hints[0].endswith("test_cifar10.txt")
    assert all("test.txt" not in hint for hint in hints)


def test_missing_path_hint_walks_up_to_real_ancestor_on_wrong_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    # Real layout: images live under data/images_classic, not data/images.
    (tmp_path / "data" / "images_classic" / "cifar10").mkdir(parents=True)
    (tmp_path / "data" / "benchmark_imglist").mkdir(parents=True)
    monkeypatch.setattr(module, "WORKDIR", tmp_path)
    context = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/workspace/data/images/cifar10/cifar10/test/airplane/0298.png'"
    )

    hints = _missing_path_hints(context)

    # The bogus parent doesn't exist; the nearest real ancestor is data/, whose
    # actual contents ground the agent toward images_classic/.
    assert any("images_classic" in hint for hint in hints)
    assert all(hint.startswith("data/") for hint in hints)


def test_round_code_endorsement_requires_all_three_signals(tmp_path: Path) -> None:
    # Freezing a repair round's code needs run_ok AND contract_passes AND a
    # Reviewer PASS — any one disputing keeps it editable.
    review = tmp_path / "review_report.md"
    passing = tmp_path / "pass.md"
    passing.write_text("Matches repository semantics.\nREVIEW_STATUS: PASS\n")

    # Regression 026: exit-0, contract passes, but the Reviewer flagged the sign.
    review.write_text("The EBO energy sign looks inverted.\nREVIEW_STATUS: REPAIR_REQUIRED\n")
    assert not _round_code_is_endorsed(True, True, review)

    # Regression 028: exit-0 and Reviewer PASS, but the deterministic contract
    # still reports a mismatch (e.g. a short TinyImageNet count) → not endorsed.
    assert not _round_code_is_endorsed(True, False, passing)

    # Failed execution is never endorsed, regardless of the rest.
    assert not _round_code_is_endorsed(False, True, passing)

    # All three agree → frozen.
    assert _round_code_is_endorsed(True, True, passing)
    # Missing review file fails closed.
    assert not _round_code_is_endorsed(True, True, tmp_path / "missing.md")


def test_repair_loop_stops_once_contract_passes_regardless_of_reviewer() -> None:
    # Regression for 029/030: once the deterministic contract fully passes, the
    # loop must stop so a paranoid Reviewer cannot drive a repair that breaks an
    # already-validated result.
    from run_openood_multi_rag import _repair_loop_should_continue

    assert not _repair_loop_should_continue(_public_contract_passes(_session(87.58, 87.58)))
    # An inverted (below-chance) result still fails the contract → keep repairing.
    assert _repair_loop_should_continue(_public_contract_passes(_session(12.42, 12.42)))


def test_review_status_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "review.md"
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: REPAIR_REQUIRED\n")
    assert _review_requires_repair(report)
    report.write_text("REVIEW_STATUS: PASS\n")
    assert not _review_requires_repair(report)


def test_dynamic_rag_query_is_generated_from_error_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "config.yml").write_text("data_root: data/images_classic\n")
    artifacts = tmp_path / "artifacts"
    query = "resolve FileNotFoundError benchmark data path"
    report = "Grounded path audit. " + ("x" * 310) + "\nREVIEW_STATUS: REPAIR_REQUIRED"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": query})]),
        Reply("", [ToolCall("s1", "submit_review", {"content": report})]),
    ])
    rag_llm = ScriptedLLM([])
    llms = iter([role_llm, rag_llm, ScriptedLLM([])])

    monkeypatch.setattr(module, "WORKDIR", workspace)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifacts)
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
    import run_openood_multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    report = "Grounded handoff. " + ("x" * 310)
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "find official evaluation entry"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM([Reply(report)])])

    monkeypatch.setattr(module, "WORKDIR", workspace)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifacts)
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module,
        "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )

    role, rag = _dynamic_rag_role(
        name="navigator_test",
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


def _synthesis_harness(tmp_path, monkeypatch, synthesis_replies, validator):
    import run_openood_multi_rag as module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifacts = tmp_path / "artifacts"
    role_llm = ScriptedLLM([
        Reply("", [ToolCall("q1", "search_repo", {"query": "find the official evaluation entry"})]),
    ])
    llms = iter([role_llm, ScriptedLLM([]), ScriptedLLM(synthesis_replies)])
    monkeypatch.setattr(module, "WORKDIR", workspace)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifacts)
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module, "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )
    with pytest.raises(RuntimeError, match="failed to synthesize"):
        _dynamic_rag_role(
            name="synth",
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
    assert "essentially identical" in transcript


def test_synthesis_escalates_on_repeated_validation_error(tmp_path: Path, monkeypatch) -> None:
    def always_fail(content: str) -> str:
        raise ValueError("artifact must define an evaluation entry")

    # Two materially different candidates that fail with the SAME error.
    transcript = _synthesis_harness(
        tmp_path, monkeypatch, [Reply("a" * 400), Reply("b" * 400)], always_fail
    )
    assert "SAME error" in transcript


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
print("REPRO_RESULT " + json.dumps({}))
"""
    with pytest.raises(ValueError, match="fixed model/CLI contract"):
        _validate_code(code)


def test_code_validator_rejects_syntax_error_before_execution() -> None:
    with pytest.raises(ValueError, match="syntactically valid"):
        _validate_code(_valid_code() + "\nif True print('broken')\n")


def test_code_validator_allows_forbidden_names_in_comments_and_strings() -> None:
    # The 024-class false positive: forbidden names mentioned in a comment, a
    # docstring, or a plain (non-call) string are documentation, not use.
    code = (
        '"""This module deliberately avoids openood.evaluation_api and never '
        'parses checkpoint config.yml files."""\n'
        + _valid_code()
        + "\n# It replicates TestStandardPreProcessor logic without importing"
        + " that class.\n"
        + 'note = "do not pass --checkpoint_root; use --root instead"\n'
        + 'doc = "ImglistDataset is reused from the repository, not redefined"\n'
    )
    assert "TestStandardPreProcessor" in code  # the name is present as prose...
    _validate_code(code)  # ...but validation passes because it is never used


def test_code_validator_rejects_real_forbidden_usage_at_ast_level() -> None:
    # Real import of a forbidden name.
    with pytest.raises(ValueError, match="forbidden import of 'TestStandardPreProcessor'"):
        _validate_code(
            _valid_code()
            + "\nfrom openood.preprocessors import TestStandardPreProcessor\n"
        )
    # Real instantiation/call of a forbidden name.
    with pytest.raises(ValueError, match="forbidden instantiation 'TestStandardPreProcessor"):
        _validate_code(_valid_code() + "\npre = TestStandardPreProcessor(cfg)\n")
    # Real re-implementation of a repository class.
    with pytest.raises(ValueError, match="forbidden re-implementation 'class ImglistDataset'"):
        _validate_code(_valid_code() + "\nclass ImglistDataset:\n    pass\n")
    # Forbidden literal actually passed to a call (parsing checkpoint config).
    with pytest.raises(ValueError, match="forbidden call argument 'config.yml'"):
        _validate_code(_valid_code() + "\ncfg = load_config(open('s0/config.yml'))\n")
    # Forbidden CLI flag actually registered with argparse.
    with pytest.raises(ValueError, match="forbidden call argument '--checkpoint_root'"):
        _validate_code(_valid_code() + "\nparser.add_argument('--checkpoint_root')\n")


def test_normalization_diagnostics_resolve_function_local_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    monkeypatch.setattr(module, "WORKDIR", tmp_path)
    # Wrong normalization assigned inside a function body (not module scope).
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "def build():\n"
        + "    mean = [0.4914, 0.4822, 0.4465]\n"
        + "    std = [0.2023, 0.1994, 0.201]\n"
        + "    return transforms.Normalize(mean, std)\n"
    )
    with pytest.raises(ValueError, match="normalization mismatch"):
        _validate_code(code)


def test_normalization_diagnostics_allow_repo_dict_subscript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    monkeypatch.setattr(module, "WORKDIR", tmp_path)
    # Using the repository normalization_dict directly is correct → not flagged.
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "from openood.preprocessors.transform import normalization_dict\n"
        + "mean, std = normalization_dict['cifar10']\n"
        + "norm = transforms.Normalize(mean, std)\n"
    )
    assert _validate_code(code)  # no normalization mismatch raised


def test_composite_rejects_obviously_fabricated_aggregate() -> None:
    # Components average to ~87 but the reported aggregate is 99 → rejected.
    assert not _public_contract_passes(_session(99.0, 87.0))


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
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

    transform = tmp_path / "openood" / "preprocessors" / "transform.py"
    transform.parent.mkdir(parents=True)
    transform.write_text(
        "normalization_dict = {'cifar10': "
        "[[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]]}\n"
    )
    monkeypatch.setattr(module, "WORKDIR", tmp_path)
    code = (
        _valid_code()
        + "\nfrom torchvision import transforms\n"
        + "transform = transforms.Normalize("
        + "mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])\n"
    )

    with pytest.raises(ValueError, match="normalization mismatch"):
        _validate_code(code)


def _valid_code() -> str:
    return """
import json
from openood.networks import ResNet18_32x32
from torch.utils.data import DataLoader
root_flag = "--root"
print("REPRO_RESULT " + json.dumps({}))
"""


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

    patched = _apply_code_patch(code_path, payload)

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
        _apply_code_patch(code_path, ambiguous)

    whole_file = json.dumps({
        "edits": [{"old": code_path.read_text(), "new": _valid_code()}],
        "rationale": "rewrite",
    })
    with pytest.raises(ValueError, match="too much"):
        _apply_code_patch(code_path, whole_file)


def test_code_patch_shows_closest_real_code_when_old_text_missing(tmp_path: Path) -> None:
    # Regression for attempt 027: repairs repeatedly failed because the patch
    # `old` text did not exist in the file (stale/paraphrased). The error should
    # now surface the closest ACTUAL lines so the next attempt copies exact code.
    code_path = tmp_path / "eval_ebo.py"
    code_path.write_text(
        _valid_code()
        + "\nnum_samples = len(loader.dataset)\n"
        + "datasets = {'cifar100': num_samples}\n"
    )
    # The agent patches against a paraphrased line that isn't literally present.
    stale = json.dumps({
        "edits": [{"old": "datasets = {'cifar100': n_samples}", "new": "datasets = {'cifar100': count}"}],
        "rationale": "fix the dataset count source",
    })
    with pytest.raises(ValueError) as exc:
        _apply_code_patch(code_path, stale)
    message = str(exc.value)
    assert "was not found" in message
    assert "Closest actual code" in message
    assert "datasets = {'cifar100': num_samples}" in message  # the real line is shown


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
            required_change_terms={"datasets", "cifar100", "tin", "len("},
        )


def test_dynamic_repair_role_submits_structured_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_openood_multi_rag as module

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
    monkeypatch.setattr(module, "WORKDIR", workspace)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifacts)
    monkeypatch.setattr(module, "ChatLLM", lambda: next(llms))
    monkeypatch.setattr(
        module,
        "search_repo",
        lambda query, root, llm, **kwargs: "Most relevant files:\n",
    )

    role, _ = _dynamic_rag_role(
        name="repair_test",
        session=Session(workspace),
        instruction="Search and patch.",
        context="Execution failed because broken is true.",
        output_path=code_path,
        submit_name="submit_patch",
        submit_description="Submit patch.",
        validator=lambda payload: _apply_code_patch(code_path, payload),
        trigger="execution_error_and_reviewer_finding",
        submit_schema=_patch_tool("submit_patch", "Submit patch."),
        submission_adapter=lambda arguments: json.dumps(arguments),
    )

    assert "broken = False" in code_path.read_text()
    assert role["tool_counts"] == {"search_repo": 1, "submit_patch": 1}
    assert role["submission_trace"] == "repair_test_submission.json"
    saved = json.loads((artifacts / "repair_test_submission.json").read_text())
    assert saved["edits"][0]["old"] == "broken = True"
