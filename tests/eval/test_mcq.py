"""Tests for MCQ evaluator."""

from __future__ import annotations

from rllm.experimental.eval.types import (
    Evaluator,
    MCQEvaluator,
)
from rllm.types import Episode

# ---------------------------------------------------------------------------
# MCQEvaluator
# ---------------------------------------------------------------------------


class TestMCQEvaluator:
    def test_correct_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "B"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B", "data_source": "test"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_answer_is_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "C"}
        ep = Episode(artifacts={"answer": "The answer is C"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_answer_is_colon_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "D"}
        ep = Episode(artifacts={"answer": "After analyzing, answer: D"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_bold_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "I think **B** is correct"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_parenthesized_pattern(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "The correct option is (A)"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_empty_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_no_ground_truth(self):
        evaluator = MCQEvaluator()
        task = {}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_lowercase_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "b"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_ten_option_answer(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "J"}
        ep = Episode(artifacts={"answer": "J"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_signals_present(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "A"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert len(result.signals) > 0
        assert result.signals[0].name == "accuracy"

    def test_metadata_contains_answers(self):
        evaluator = MCQEvaluator()
        task = {"ground_truth": "B"}
        ep = Episode(artifacts={"answer": "A"})
        result = evaluator.evaluate(task, ep)
        assert result.metadata["model_answer"] == "A"
        assert result.metadata["expected"] == "B"

    def test_is_evaluator(self):
        evaluator = MCQEvaluator()
        assert isinstance(evaluator, Evaluator)


# ---------------------------------------------------------------------------
# MCQEvaluator._extract_choice_letter edge cases
# ---------------------------------------------------------------------------


class TestExtractChoiceLetter:
    def test_single_letter(self):
        assert MCQEvaluator._extract_choice_letter("A") == "A"

    def test_single_letter_lowercase(self):
        assert MCQEvaluator._extract_choice_letter("c") == "C"

    def test_verbose_response(self):
        assert MCQEvaluator._extract_choice_letter("After careful analysis, the answer is B") == "B"

    def test_no_letter(self):
        assert MCQEvaluator._extract_choice_letter("No valid answer here") == ""

    def test_empty(self):
        assert MCQEvaluator._extract_choice_letter("") == ""

    def test_letter_in_word(self):
        # "A" as a standalone word (article) — should match as fallback
        result = MCQEvaluator._extract_choice_letter("A long explanation about biology")
        assert result == "A"

    def test_answer_with_parentheses(self):
        assert MCQEvaluator._extract_choice_letter("The answer is (D)") == "D"
