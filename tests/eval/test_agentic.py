"""Tests for agentic benchmarks: BFCL and LLM judge reward functions.

After the inlining refactor, the per-task scorers live as plain
``evaluate(task, episode)`` functions in ``rllm.eval.reward_fns.*``.
``load_evaluator(name)`` returns a thin wrapper that auto-adapts the
legacy dict-call form used by these tests.
"""

from __future__ import annotations

import json

from rllm.eval.evaluator_loader import load_evaluator
from rllm.eval.reward_fns.bfcl import _compare_function_calls
from rllm.types import Episode, Evaluator

# ---------------------------------------------------------------------------
# bfcl_reward_fn
# ---------------------------------------------------------------------------


class TestBFCLEvaluator:
    def test_correct_function_call(self):
        evaluator = load_evaluator("bfcl_reward_fn")
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
        evaluator = load_evaluator("bfcl_reward_fn")
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
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {
            "ground_truth": [json.dumps({"name": "func", "arguments": {}})],
        }
        ep = Episode(artifacts={"answer": "", "tool_calls": []})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_no_ground_truth(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        task = {"ground_truth": []}
        ep = Episode(artifacts={"answer": "", "tool_calls": [{"name": "f", "arguments": "{}"}]})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True

    def test_is_evaluator(self):
        evaluator = load_evaluator("bfcl_reward_fn")
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = load_evaluator("bfcl_reward_fn")
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
# llm_judge_reward_fn
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    def test_no_rubric_fallback(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # No rubric → passes if response is non-empty
        assert result.is_correct is True
        assert result.metadata.get("reason") == "no_rubric_available"

    def test_empty_answer_no_rubric(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_is_evaluator(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        assert isinstance(evaluator, Evaluator)

    def test_signals_present(self):
        evaluator = load_evaluator("llm_judge_reward_fn")
        task = {"question": "Hello"}
        ep = Episode(artifacts={"answer": "Hi"})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "judge_score" for s in result.signals)

    def test_with_rubric_no_judge(self):
        evaluator = load_evaluator("llm_judge_reward_fn")  # No judge_base_url
        task = {"question": "Hello", "rubric": "Should greet politely"}
        ep = Episode(artifacts={"answer": "Hi there!"})
        result = evaluator.evaluate(task, ep)
        # Falls back since no judge available
        assert result.metadata.get("reason") == "judge_unavailable_fallback"
