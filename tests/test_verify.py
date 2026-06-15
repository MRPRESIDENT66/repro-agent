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
    # A self-contained inline eval (python -c) that REALLY loads data and predicts
    # (real call nodes, not comment markers) must be accepted.
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    cmd = (
        'python -c "import datasets; d = datasets.load_dataset(); '
        "pred = logits.argmax(1); print('REPRO_RESULT ...')\""
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
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "tools" / "test.py").write_text("# the repo's real eval entry\n")
    (tmp_path / "wrapper.py").write_text(
        "import subprocess, sys\n"
        "out = subprocess.run([sys.executable, 'tools/test.py', 'cfg.py', 'ckpt.pth'],\n"
        "                     capture_output=True, text=True).stdout\n"
        "print('REPRO_RESULT {\"metric\":\"top1_accuracy\",\"actual\":94.82,\"num_examples\":10000}')\n"
    )
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":94.82,"num_examples":10000}'
    verdict = verify_run(
        [_run(line, command="python wrapper.py")],
        tmp_path,
        expected=94.82,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )
    assert verdict.match


# --- Provenance attack suite: every forgery must fail closed (P0-1) -----------

def _attack(tmp_path, files: dict[str, str], command: str):
    """Write decoy files, run a forged emitting command with the RIGHT value, and
    return the verdict. Provenance is the only thing that may stop the match."""
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    line = 'REPRO_RESULT {"metric":"top1_accuracy","actual":92.60,"num_examples":10000}'
    return verify_run(
        [_run(line, command=command)],
        tmp_path,
        expected=92.60,
        tolerance=0.10,
        metric="top1_accuracy",
        expected_num_examples=10000,
    )


def test_attack_decoy_file_plus_python_c_print_fails_closed(tmp_path) -> None:
    # A legit-looking decoy script in the workspace + the value printed by a bare
    # `python -c`. The gate is bound to the emitting command, so the decoy is never
    # scanned and the print has no real eval.
    v = _attack(
        tmp_path,
        {"decoy.py": "import datasets\nd = datasets.load_dataset()\nx.argmax(1)\n# REPRO_RESULT\n"},
        command="python -c \"print('REPRO_RESULT {...}')\"",
    )
    assert not v.match and v.reason == "no_eval_provenance"


def test_attack_comment_markers_fail_closed(tmp_path) -> None:
    # The emitting script carries the marker words only in comments/strings.
    v = _attack(
        tmp_path,
        {"eval.py": "# load_dataset argmax over CIFAR\nx = 'argmax load_dataset'\n"
                    "print('REPRO_RESULT {...}')\n"},
        command="python eval.py",
    )
    assert not v.match and v.reason == "no_eval_provenance"


def test_attack_hardcoded_print_fails_closed(tmp_path) -> None:
    v = _attack(
        tmp_path,
        {"eval.py": "print('REPRO_RESULT {\"metric\":\"top1_accuracy\","
                    "\"actual\":92.60,\"num_examples\":10000}')\n"},
        command="python eval.py",
    )
    assert not v.match and v.reason == "no_eval_provenance"


def test_attack_fake_wrapper_without_subprocess_fails_closed(tmp_path) -> None:
    # Mentions the entry + checkpoint as plain strings but never shells out.
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "test.py").write_text("# entry\n")
    v = _attack(
        tmp_path,
        {"eval.py": "entry = 'tools/test.py'\nckpt = 'ckpt.pth'\n"
                    "print('REPRO_RESULT {...}')\n"},
        command="python eval.py",
    )
    assert not v.match and v.reason == "no_eval_provenance"


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
