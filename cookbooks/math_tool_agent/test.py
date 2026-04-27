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
    assert "Error" in _safe_eval("a" * 201)


def test_safe_eval_sqrt():
    assert _safe_eval("sqrt(9)") == "3"
    assert _safe_eval("sqrt(64 + 225)") == "17"


def test_safe_eval_abs():
    assert _safe_eval("abs(-5)") == "5"
    assert _safe_eval("abs(3.14 - 2.71)") == "0.43"


def test_safe_eval_combinatorics():
    assert _safe_eval("comb(5, 2)") == "10"
    assert _safe_eval("binom(10, 3)") == "120"
    assert _safe_eval("factorial(5)") == "120"


def test_safe_eval_constants():
    assert _safe_eval("pi") == "3.141593"
    assert _safe_eval("e") == "2.718282"


def test_safe_eval_trig():
    # cos(0) = 1
    assert _safe_eval("cos(0)") == "1"
    # sin(pi/2) = 1
    assert _safe_eval("sin(pi / 2)") == "1"


def test_safe_eval_nested_functions():
    # 2 * sqrt(2) - 2
    out = _safe_eval("2 * sqrt(2) - 2")
    assert out.startswith("0.828")


def test_safe_eval_rejects_attribute_access():
    assert "Error" in _safe_eval("(1).bit_length()")
    assert "Error" in _safe_eval("math.sqrt(4)")  # 'math' not in whitelist


def test_safe_eval_rejects_comparison():
    assert "Error" in _safe_eval("1 < 2")


def test_safe_eval_rejects_unknown_name():
    assert "unknown name" in _safe_eval("foo(2)")


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
