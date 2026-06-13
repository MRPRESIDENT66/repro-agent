"""The standalone single-line verifier + the MCP server's tool surface.

The verifier is the deterministic core; the MCP layer is a thin wrapper, so we
test the logic on the pure function and only assert that the server registers
the three tools and that its verify tool delegates correctly.
"""

from __future__ import annotations

import pytest

from verify.check import verify_evidence_line

_LINE = '   REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000}'


def test_matches_within_tolerance() -> None:
    v = verify_evidence_line(_LINE, expected=92.6, tolerance=0.1,
                             metric="top1_accuracy", expected_num_examples=10000)
    assert v.match and v.actual == 92.6 and v.abs_diff == 0.0 and v.reason is None


def test_outside_tolerance_fails_with_reason() -> None:
    v = verify_evidence_line(_LINE, expected=90.0, tolerance=0.1,
                             metric="top1_accuracy", expected_num_examples=10000)
    assert not v.match and v.actual == 92.6 and v.reason == "outside_tolerance"


def test_num_examples_mismatch_fails_closed() -> None:
    # a 100-example "eval" cannot pass off as the full 10000-example claim
    v = verify_evidence_line(
        '{f}'.replace('{f}', _LINE.replace("10000", "100")),
        expected=92.6, tolerance=0.1, metric="top1_accuracy", expected_num_examples=10000)
    assert not v.match and v.reason == "num_examples_mismatch"


def test_metric_and_target_mismatch() -> None:
    v = verify_evidence_line(_LINE, expected=92.6, tolerance=0.1,
                             metric="f1", expected_num_examples=10000)
    assert not v.match and v.reason == "metric_mismatch"

    line_t = '''REPRO_RESULT {"metric":"top1_accuracy","actual":92.6,"num_examples":10000,"target":"resnet20"}'''
    v2 = verify_evidence_line(line_t, expected=92.6, tolerance=0.1, metric="top1_accuracy",
                              expected_num_examples=10000, target="resnet56")
    assert not v2.match and v2.reason == "target_mismatch"


@pytest.mark.parametrize("bad", ["not even structured", "REPRO_RESULT {nope}", "REPRO_RESULT {}"])
def test_garbage_fails_closed(bad: str) -> None:
    v = verify_evidence_line(bad, expected=92.6, tolerance=0.1,
                             metric="top1_accuracy", expected_num_examples=10000)
    assert not v.match and v.reason in {"not_a_repro_result_line", "malformed_evidence"}


def test_mcp_server_registers_three_tools() -> None:
    import serve_mcp

    names = {t.name for t in serve_mcp.mcp._tool_manager.list_tools()}
    assert names == {"verify_evidence_line", "navigate_repo", "reproduce_artifact"}


def test_mcp_verify_tool_delegates() -> None:
    import serve_mcp

    out = serve_mcp.verify_evidence_line(
        _LINE, expected=92.6, tolerance=0.1, metric="top1_accuracy", num_examples=10000)
    assert out["match"] is True and out["actual"] == 92.6
