from agent.failure import classify_failure
from exec.session import RunResult


class _Session:
    def __init__(self, stderr: str = "", stdout: str = "", command: str = "python eval.py"):
        self.transcript = [
            RunResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_code=1,
                timed_out=False,
                duration_s=0.0,
            )
        ]


def test_classifies_import_error_with_probe_hint():
    failure = classify_failure(
        session=_Session(stderr="ModuleNotFoundError: No module named 'openood'"),
        diagnostics=[],
    )

    assert failure.kind == "import_error"
    assert failure.probe_hint == "import_smoke:openood"


def test_classifies_missing_artifact_from_diagnostics():
    failure = classify_failure(
        session=_Session(stdout="finished"),
        diagnostics=["The required public result artifact is missing after execution (missing: ['predictions.json'])."],
    )

    assert failure.kind == "missing_artifact"
    assert "artifact" in failure.next_action


def test_classifies_semantic_mismatch():
    failure = classify_failure(
        session=_Session(stdout="ran"),
        diagnostics=["outside_tolerance: verifier recomputed lower value"],
    )

    assert failure.kind == "semantic_mismatch"
