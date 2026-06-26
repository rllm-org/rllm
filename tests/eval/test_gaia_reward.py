"""Tests for the GAIA reward function, the GAIA row transform, and registry wiring.

The scorer mirrors the official GAIA `question_scorer` (number / list / string
normalization). `evaluate` is exercised through `load_evaluator` so the dict-call
adapter path is covered too.
"""

from __future__ import annotations

import json
from pathlib import Path

from rllm.data.transforms import gaia_transform
from rllm.eval.evaluator_loader import load_evaluator
from rllm.eval.reward_fns.gaia import question_scorer
from rllm.types import Episode


# ---------------------------------------------------------------------------
# question_scorer (official GAIA quasi-exact-match)
# ---------------------------------------------------------------------------
class TestQuestionScorer:
    def test_string_exact(self):
        assert question_scorer("Paris", "Paris") is True

    def test_string_normalizes_case_space_punct(self):
        assert question_scorer("  the   PARIS! ", "the paris") is True

    def test_string_mismatch(self):
        assert question_scorer("London", "Paris") is False

    def test_number_value_match_ignoring_separators(self):
        assert question_scorer("$1,000.00", "1000") is True
        assert question_scorer("17%", "17") is True

    def test_number_mismatch(self):
        assert question_scorer("42", "43") is False

    def test_number_garbage_answer_is_wrong_not_crash(self):
        # non-numeric answer against a numeric gt -> inf sentinel -> not equal
        assert question_scorer("not a number", "5") is False

    def test_list_elementwise_match(self):
        assert question_scorer("apple, banana, cherry", "apple, banana, cherry") is True

    def test_list_wrong_length(self):
        assert question_scorer("apple, banana", "apple, banana, cherry") is False

    def test_list_mixed_number_and_string(self):
        assert question_scorer("1, two, 3", "1, two, 3") is True
        assert question_scorer("1, two, 4", "1, two, 3") is False


# ---------------------------------------------------------------------------
# gaia_transform
# ---------------------------------------------------------------------------
class TestGaiaTransform:
    def test_text_question(self):
        out = gaia_transform({"Question": "What is 2+2?", "Final answer": "4", "Level": 1, "file_name": ""})
        assert out == {"question": "What is 2+2?", "ground_truth": "4", "level": 1, "data_source": "gaia"}

    def test_skips_file_attached_tasks(self):
        out = gaia_transform({"Question": "Read the file.", "Final answer": "x", "file_name": "data.xlsx"})
        assert out is None

    def test_skips_empty_question(self):
        assert gaia_transform({"Question": "  ", "Final answer": "x"}) is None


# ---------------------------------------------------------------------------
# evaluate() via load_evaluator (dict-call adapter path)
# ---------------------------------------------------------------------------
class TestGaiaEvaluate:
    def _ev(self):
        return load_evaluator("gaia_reward_fn")

    def test_correct(self):
        out = self._ev().evaluate({"ground_truth": "Paris"}, Episode(artifacts={"answer": "Paris"}))
        assert out.is_correct is True
        assert out.reward == 1.0

    def test_incorrect(self):
        out = self._ev().evaluate({"ground_truth": "Paris"}, Episode(artifacts={"answer": "London"}))
        assert out.is_correct is False
        assert out.reward == 0.0

    def test_strips_final_answer_prefix_and_normalizes_number(self):
        # answer after the prefix is "$1,000"; GAIA strips $/, in the number path.
        # (NB: a *ground truth* with a comma would be parsed as a list, not a number.)
        ep = Episode(artifacts={"answer": "Let me think... FINAL ANSWER: $1,000"})
        out = self._ev().evaluate({"ground_truth": "1000"}, ep)
        assert out.is_correct is True

    def test_missing_ground_truth(self):
        out = self._ev().evaluate({}, Episode(artifacts={"answer": "Paris"}))
        assert out.is_correct is False
        assert "error" in out.metadata


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------
def test_gaia_registered_in_catalog():
    catalog = json.loads((Path(__file__).parents[2] / "rllm" / "registry" / "datasets.json").read_text())
    entry = catalog["datasets"]["gaia"]
    assert entry["reward_fn"] == "gaia_reward_fn"
    assert entry["transform"] == "rllm.data.transforms:gaia_transform"
    assert entry["eval_split"] == "validation"
    assert entry["gated"] is True
