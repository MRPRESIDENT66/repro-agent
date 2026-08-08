from agent.orchestration.failure import classify_failure
from exec.session import RunResult


class _Session:
    def __init__(
        self,
        stderr: str = "",
        stdout: str = "",
        command: str = "python eval.py",
        timed_out: bool = False,
    ):
        self.transcript = [
            RunResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_code=1,
                timed_out=timed_out,
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


def test_classifies_multiprocessing_pickling_before_missing_artifact():
    failure = classify_failure(
        session=_Session(
            stderr="AttributeError: Can't get local object 'build.<locals>.<lambda>'"
        ),
        diagnostics=[
            "The required public result artifact is missing after execution "
            "(missing: ['predictions.json'])."
        ],
    )

    assert failure.kind == "multiprocessing_serialization"
    assert "num_workers=0" in failure.next_action


def test_classifies_attribute_error_before_unrelated_pickle_warning():
    failure = classify_failure(
        session=_Session(
            stderr=(
                "warning: entry = pickle.load(f)\n"
                "AttributeError: 'AutoAttack' object has no attribute 'apgd_dlr'"
            )
        ),
        diagnostics=[],
    )

    assert failure.kind == "api_mismatch"
    assert failure.probe_hint == "python_source:<object named in traceback>"


def test_classifies_timeout_before_missing_artifact_diagnostic():
    failure = classify_failure(
        session=_Session(timed_out=True),
        diagnostics=[
            "The required public result artifact is missing after execution "
            "(missing: ['predictions.json'])."
        ],
    )

    assert failure.kind == "timeout"
    assert "unrequested algorithms" in failure.next_action
    assert "without reducing requested sample coverage" in failure.next_action
