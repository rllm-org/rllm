"""Tests for the math tool agent cookbook."""

from evaluator import math_tool_evaluator
from math_tool_agent import _extract_answer, _safe_eval

from rllm.types import Episode, Step, Trajectory

# -- Calculator tests ----------------------------------------------------------


def test_safe_eval_basic():
    assert _safe_eval("2 + 3") == "5"
    assert _safe_eval("34 * 17") == "578"
    assert _safe_eval("100 - 37") == "63"


def test_safe_eval_float():
    assert _safe_eval("10 / 4") == "2.5"
    assert _safe_eval("10 / 2") == "5"  # returns int when exact


def test_safe_eval_parentheses():
    assert _safe_eval("(10 + 5) * 3") == "45"


def test_safe_eval_power():
    assert _safe_eval("2 ** 10") == "1024"
    assert _safe_eval("3 ** 2") == "9"


def test_safe_eval_modulo():
    assert _safe_eval("17 % 5") == "2"


def test_safe_eval_rejects_invalid():
    assert "Error" in _safe_eval("import os")
    assert "Error" in _safe_eval("__import__('os')")
    assert "Error" in _safe_eval("a" * 101)


# -- Answer extraction tests ---------------------------------------------------


def test_extract_answer_boxed():
    assert _extract_answer("Therefore \\boxed{42}") == "42"


def test_extract_answer_tag():
    assert _extract_answer("blah <answer>42</answer> blah") == "42"


def test_extract_answer_hashes():
    assert _extract_answer("So the total is 42.\n#### 42") == "42"


def test_extract_answer_natural():
    assert _extract_answer("The final answer is 42.") == "42"


def test_extract_answer_last_number():
    assert _extract_answer("We get 10 + 32 = 42") == "42"


def test_extract_answer_with_think():
    assert _extract_answer("<think>Let me think... 6*7=42</think>#### 42") == "42"


def test_extract_answer_commas():
    assert _extract_answer("The answer is 1,234") == "1234"


# -- Evaluator tests -----------------------------------------------------------


def test_evaluator_correct():
    task = {"question": "What is 3 + 7?", "ground_truth": "10"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step(action="<answer>10</answer>")])],
        artifacts={"answer": "10"},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is True
    assert result.reward == 1.0


def test_evaluator_wrong():
    task = {"question": "What is 3 + 7?", "ground_truth": "10"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step(action="<answer>9</answer>")])],
        artifacts={"answer": "9"},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is False
    assert result.reward == 0.0


def test_evaluator_no_answer():
    task = {"question": "What is 3 + 7?", "ground_truth": "10"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step(action="I don't know")])],
        artifacts={"answer": ""},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is False
    assert result.reward == 0.0


def test_evaluator_float_answer():
    task = {"question": "Split $10 among 4 people equally.", "ground_truth": "2.5"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step()])],
        artifacts={"answer": "2.5"},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is True


def test_evaluator_latex_fraction():
    task = {"question": "What is 1/2?", "ground_truth": "\\frac{1}{2}"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step()])],
        artifacts={"answer": "0.5"},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is True


def test_evaluator_comma_number():
    task = {"question": "How much?", "ground_truth": "1234"}
    episode = Episode(
        trajectories=[Trajectory(name="solver", steps=[Step()])],
        artifacts={"answer": "1,234"},
    )
    result = math_tool_evaluator.evaluate(task, episode)
    assert result.is_correct is True
