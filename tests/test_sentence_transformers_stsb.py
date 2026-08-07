import json

import pytest

from agent.contracts import make_generic_code_validator
from evals.oracles import sentence_transformers_stsb as oracle


def _write_predictions(path, similarities) -> None:
    rows = [
        {"id": index, "similarity": value}
        for index, value in enumerate(similarities)
    ]
    (path / "predictions.json").write_text(json.dumps(rows))


def test_spearman_handles_ties() -> None:
    assert oracle._spearman(
        [0.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 3.0]
    ) == pytest.approx(1.0)
    assert oracle._spearman(
        [0.0, 1.0, 2.0], [3.0, 2.0, 1.0]
    ) == pytest.approx(-1.0)


def test_prediction_contract_rejects_wrong_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oracle, "N_EXAMPLES", 2)
    rows = [{"id": 1, "similarity": 0.2}, {"id": 0, "similarity": 0.3}]
    (tmp_path / "predictions.json").write_text(json.dumps(rows))

    assert oracle._load_similarities(tmp_path) is None


def test_recompute_uses_private_gold(tmp_path, monkeypatch) -> None:
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps([0.0, 1.0, 2.0]))
    monkeypatch.setattr(oracle, "GOLD_SCORES", gold_path)
    monkeypatch.setattr(oracle, "N_EXAMPLES", 3)
    _write_predictions(tmp_path, [0.1, 0.2, 0.3])

    score, count = oracle._recompute(tmp_path)
    assert score == pytest.approx(1.0)
    assert count == 3


def test_cli_output_path_is_valid_without_hardcoded_filename() -> None:
    validator = make_generic_code_validator(oracle.make_config("contract_test"))
    code = """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.output, "w") as handle:
    json.dump([], handle)
"""

    assert validator(code) == code.strip() + "\n"
