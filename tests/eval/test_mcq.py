"""Tests for the MCQ reward function and answer extraction.

The class-based ``MCQEvaluator`` was inlined into
``rllm.eval.reward_fns.mcq`` as a plain ``evaluate(task, episode)``
function. ``load_evaluator`` returns an adapter that accepts the
legacy dict-call form.
"""

from __future__ import annotations

from rllm.eval.evaluator_loader import load_evaluator
from rllm.eval.reward_fns.mcq import _extract_choice_letter
from rllm.types import Episode, Evaluator


def _mcq():
    return load_evaluator("mcq_reward_fn")


# ---------------------------------------------------------------------------
# mcq_reward_fn
# ---------------------------------------------------------------------------


class TestMCQEvaluator:
    def test_correct_answer(self):
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "B"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_answer(self):
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "A"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_answer_is_pattern(self):
        task = {"ground_truth": "C"}
        ep = Episode(artifacts={"answer": "The answer is C"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_answer_is_colon_pattern(self):
        task = {"ground_truth": "D"}
        ep = Episode(artifacts={"answer": "After analyzing, answer: D"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_bold_pattern(self):
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "I think **B** is correct"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_parenthesized_pattern(self):
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "The correct option is (A)"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_empty_answer(self):
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": ""})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_no_ground_truth(self):
        task = {}
        ep = Episode(artifacts={"answer": "A"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is False

    def test_lowercase_answer(self):
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "b"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_ten_option_answer(self):
        task = {"ground_truth": "J"}
        ep = Episode(artifacts={"answer": "J"})
        result = _mcq().evaluate(task, ep)
        assert result.is_correct is True

    def test_signals_present(self):
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "A"})
        result = _mcq().evaluate(task, ep)
        assert len(result.signals) > 0
        assert result.signals[0].name == "accuracy"

    def test_metadata_contains_answers(self):
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "A"})
        result = _mcq().evaluate(task, ep)
        assert result.metadata["model_answer"] == "A"
        assert result.metadata["expected"] == "B"

    def test_is_evaluator(self):
        assert isinstance(_mcq(), Evaluator)


# ---------------------------------------------------------------------------
# _extract_choice_letter (free function)
# ---------------------------------------------------------------------------


class TestExtractChoiceLetter:
    def test_single_letter(self):
        assert _extract_choice_letter("A") == "A"

    def test_single_letter_lowercase(self):
        assert _extract_choice_letter("c") == "C"

    def test_verbose_response(self):
        assert _extract_choice_letter("After careful analysis, the answer is B") == "B"

    def test_no_letter(self):
        assert _extract_choice_letter("No valid answer here") == ""

    def test_empty(self):
        assert _extract_choice_letter("") == ""

    def test_letter_in_word(self):
        # "A" as a standalone word (article) — should match as fallback
        assert _extract_choice_letter("A long explanation about biology") == "A"

    def test_answer_with_parentheses(self):
        assert _extract_choice_letter("The answer is (D)") == "D"
