"""Blind verification protocol: only structured command evidence can match."""

from __future__ import annotations

from exec.session import RunResult
from verify.check import verify_run, verify_transcript


def _run(stdout: str, *, ok: bool = True, command: str = "python eval.py") -> RunResult:
    return RunResult(
        command=command,
        stdout=stdout,
        stderr="",
        exit_code=0 if ok else 1,
        timed_out=False,
        duration_s=1.0,
    )


def _verify(transcript: list[RunResult]):
    return verify_transcript(
        transcript,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )


def test_final_or_unstructured_number_cannot_match() -> None:
    verdict = _verify([_run("FINAL: 92.60\nAccuracy: 92.60")])
    assert not verdict.match
    assert verdict.reason == "no_valid_structured_evidence"


def test_wrong_num_examples_cannot_match() -> None:
    verdict = _verify([
        _run('REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":100}')
    ])
    assert not verdict.match


def test_failed_command_evidence_cannot_match() -> None:
    verdict = _verify([
        _run(
            'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}',
            ok=False,
        )
    ])
    assert not verdict.match


def test_direct_echo_evidence_cannot_match() -> None:
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    verdict = _verify([_run(line, command=f"echo '{line}'")])
    assert not verdict.match


def test_valid_structured_evidence_matches() -> None:
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    verdict = _verify([_run(line)])
    assert verdict.match
    assert verdict.actual == 92.60
    assert verdict.command_index == 1
    assert verdict.num_examples == 10000
    assert verdict.evidence_line == line


def test_multi_target_evidence_must_match_target() -> None:
    transcript = [
        _run(
            'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,'
            '"num_examples":10000,"target":"resnet20"}'
        )
    ]
    verdict = verify_transcript(
        transcript,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
        target="resnet32",
    )
    assert not verdict.match


def test_structured_result_without_eval_provenance_cannot_match(tmp_path) -> None:
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    (tmp_path / "cheat.py").write_text(f"print({line!r})")
    verdict = verify_run(
        [_run(line)],
        tmp_path,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert not verdict.match
    assert verdict.reason == "no_eval_provenance"


def test_structured_result_with_eval_provenance_matches(tmp_path) -> None:
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    (tmp_path / "eval.py").write_text(
        "from datasets import load_dataset\n"
        "dataset = load_dataset('x')\n"
        "predicted = model(x).argmax(1)\n"
        f"print({line!r})\n"
    )
    verdict = verify_run(
        [_run(line)],
        tmp_path,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert verdict.match


def test_inline_eval_command_counts_as_provenance(tmp_path) -> None:
    # The eval is inline (python -c) with NO .py file — provenance must still be
    # found in the command body, else a legit inline reproduction is wrongly rejected.
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    cmd = (
        "python -c \"from datasets import load_dataset; "
        "p = model(x).argmax(1); print('REPRO_RESULT ...')\""
    )
    verdict = verify_run(
        [_run(line, command=cmd)],
        tmp_path,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert verdict.match


def test_delegating_to_repo_eval_entry_counts_as_provenance(tmp_path) -> None:
    # Clone-and-navigate: the agent runs the repo's OWN eval entry (tools/test.py)
    # against the checkpoint and parses its output. The argmax lives in the repo's
    # library code, not the agent's command — so the inline-marker heuristic would
    # false-negative this *correct* behaviour. Delegation provenance must catch it.
    (tmp_path / "repo" / "tools").mkdir(parents=True)
    (tmp_path / "repo" / "tools" / "test.py").write_text("# the repo's real eval entry\n")
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":94.82,"num_examples":10000}'
    cmd = (
        "cd repo && python -c \"import subprocess, re, sys; "
        "out = subprocess.run([sys.executable, 'tools/test.py', 'configs/r18.py', "
        "'../ckpt.pth'], capture_output=True, text=True).stdout; "
        "print('REPRO_RESULT {\\\"metric\\\":\\\"top1_accuracy\\\",...}')\""
    )
    verdict = verify_run(
        [_run(line, command=cmd)],
        tmp_path,
        expected=94.82,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert verdict.match


def test_echo_without_real_eval_still_rejected(tmp_path) -> None:
    # The gate's purpose survives the delegation extension: a bare echo of the
    # line, with no eval script on disk and no markers, must NOT pass provenance.
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":94.82,"num_examples":10000}'
    verdict = verify_run(
        [_run(line, command="echo 'REPRO_RESULT ...'")],
        tmp_path,
        expected=94.82,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert not verdict.match and verdict.reason in {"no_valid_structured_evidence", "no_eval_provenance"}
