"""Tests for agentic benchmarks: BFCL evaluator and LLM judge evaluator."""

from __future__ import annotations

import json

from rllm.experimental.eval.bfcl_evaluator import BFCLEvaluator, _compare_function_calls
from rllm.experimental.eval.llm_judge_evaluator import LLMJudgeEvaluator
from rllm.experimental.eval.types import Evaluator
from rllm.types import Episode

# ---------------------------------------------------------------------------
# BFCLEvaluator
# ---------------------------------------------------------------------------


class TestBFCLEvaluator:
    def test_correct_function_call(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(
            artifacts={
                "answer": "",
                "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Paris"}'}],
            }
        )
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_function_name(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})],
        }
        ep = Episode(
            artifacts={
                "answer": "",
                "tool_calls": [{"name": "get_time", "arguments": '{"city": "Paris"}'}],
            }
        )
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_model_calls(self):
        evaluator = BFCLEvaluator()
        task = {
            "ground_truth": [json.dumps({"name": "func", "arguments": {}})],
        }
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_ground_truth(self):
        evaluator = BFCLEvaluator()
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": [{"name": "f", "arguments": "{}"}]})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_is_evaluator(self):
        evaluator = BFCLEvaluator()
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = BFCLEvaluator()
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "ast_accuracy" for s in result.signals)


# ---------------------------------------------------------------------------
# _compare_function_calls
# ---------------------------------------------------------------------------


class TestCompareFunctionCalls:
    def test_exact_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "f", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is True

    def test_no_match(self):
        model = [{"name": "f", "arguments": {"a": 1}}]
        gt = [json.dumps({"name": "g", "arguments": {"a": 1}})]
        is_correct, _ = _compare_function_calls(model, gt)
        assert is_correct is False

    def test_empty_model(self):
        gt = [json.dumps({"name": "f", "arguments": {}})]
        is_correct, _ = _compare_function_calls([], gt)
        assert is_correct is False

    def test_empty_gt(self):
        is_correct, _ = _compare_function_calls([{"name": "f"}], [])
        assert is_correct is True


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    def test_no_rubric_fallback(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # No rubric → passes if response is non-empty
        assert result.is_correct is True
        assert result.metadata.get("reason") == "no_rubric_available"

    def test_empty_answer_no_rubric(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_is_evaluator(self):
        evaluator = LLMJudgeEvaluator()
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = LLMJudgeEvaluator()
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi"})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "judge_score" for s in result.signals)

    def test_with_rubric_no_judge(self):
        evaluator = LLMJudgeEvaluator()  # No judge_base_url
        task = {"question": "Hello", "rubric": "Should greet politely"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # Falls back since no judge available
        assert result.metadata.get("reason") == "judge_unavailable_fallback"
