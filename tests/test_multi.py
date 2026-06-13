"""Multi-agent plumbing — parallel vs serial isolation, driven by ScriptedLLM.

Faithful to the real semantics:
  * **multi** — each Reproducer is isolated (own session, own scripted LLM) and
    emits exactly ONE result line (no `target` needed); run_multi verifies it
    with `target=None`. The scripted eval carries real provenance markers and
    prints a genuine REPRO_RESULT, so the deterministic Verifier runs end to end.
  * **single** — one agent emits all N lines in one transcript, disambiguated by
    `target`.
No torch, no API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.llm import Reply, ScriptedLLM, ToolCall
from agents.multi import run_multi, run_single

# An isolated Reproducer prints ONE line, no target — matches its own sub only.
_MULTI_EVAL = """cat > eval.py <<'PYEOF'
# provenance markers: load_dataset + argmax over CIFAR
print('REPRO_RESULT {"metric":"top1_accuracy","actual":92.0,"num_examples":100}')
PYEOF"""

# The single agent prints ALL THREE, each labelled with its target.
_SINGLE_EVAL = """cat > eval.py <<'PYEOF'
# provenance markers: load_dataset + argmax over CIFAR
print('REPRO_RESULT {"metric":"top1_accuracy","actual":92.0,"num_examples":100,"target":"resnet20"}')
print('REPRO_RESULT {"metric":"top1_accuracy","actual":92.0,"num_examples":100,"target":"resnet32"}')
print('REPRO_RESULT {"metric":"top1_accuracy","actual":92.0,"num_examples":100,"target":"resnet56"}')
PYEOF"""


def _factory(eval_heredoc: str):
    def make() -> ScriptedLLM:
        return ScriptedLLM([
            Reply("", [ToolCall("c1", "bash", {"command": eval_heredoc})]),
            Reply("", [ToolCall("c2", "bash", {"command": "python3 eval.py"})]),
            Reply("", [ToolCall("c3", "finish", {"summary": "done"})]),
        ])
    return make


MANIFEST = {
    "repo": "https://github.com/example/pytorch-cifar-models",
    "dataset": {"name": "CIFAR-10 test (10000 images)", "num_examples": 100},
    "metric": "top1_accuracy",
    "tolerance": 0.1,
    "subtargets": [
        {"model": "resnet20", "expected": 92.0},
        {"model": "resnet32", "expected": 92.0},
        {"model": "resnet56", "expected": 92.0},
    ],
}


@pytest.mark.parametrize("parallel", [True, False])
def test_multi_runs_and_verifies(tmp_path: Path, parallel: bool) -> None:
    out = run_multi(MANIFEST, tmp_path, repro_py=None, make_llm=_factory(_MULTI_EVAL),
                    parallel=parallel)
    assert out["mode"] == ("multi-parallel" if parallel else "multi-serial")
    assert out["n_agents"] == 3 and out["matched"] == 3
    assert {r.model for r in out["results"]} == {"resnet20", "resnet32", "resnet56"}
    assert all(r.matched and r.actual == 92.0 for r in out["results"])
    assert out["wall_s"] >= 0 and out["max_ctx_chars"] > 0
    assert out["total_cost_yuan"] == 0  # scripted → no tokens


def test_each_agent_is_isolated(tmp_path: Path) -> None:
    # Distinct workdirs per sub-target = isolated on-disk state.
    out = run_multi(MANIFEST, tmp_path, repro_py=None, make_llm=_factory(_MULTI_EVAL),
                    parallel=True)
    workdirs = {p.name for p in tmp_path.glob("multi_*")}
    assert workdirs == {"multi_resnet20", "multi_resnet32", "multi_resnet56"}
    assert out["matched"] == 3


def test_single_shares_one_context(tmp_path: Path) -> None:
    out = run_single(MANIFEST, tmp_path, repro_py=None, make_llm=_factory(_SINGLE_EVAL))
    assert out["mode"] == "single" and out["n_agents"] == 1
    assert out["matched"] == 3  # one agent, one eval, three targeted lines verified
