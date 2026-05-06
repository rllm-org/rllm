"""Tests for the LangGraph math cookbook."""

from evaluator import _extract_answer, _extract_boxed, langgraph_math_evaluator
from langgraph_math import _safe_eval

from rllm.types import Episode, Step, Trajectory

# -- Calculator tests ----------------------------------------------------------


def test_safe_eval_basic():
    assert _safe_eval("2 + 3") == "5"
    assert _safe_eval("34 * 17") == "578"


def test_safe_eval_float():
    assert _safe_eval("10 / 4") == "2.5"
    assert _safe_eval("10 / 2") == "5"


def test_safe_eval_parentheses():
    assert _safe_eval("(10 + 5) * 3") == "45"


def test_safe_eval_power():
    assert _safe_eval("2 ** 10") == "1024"


def test_safe_eval_sqrt():
    assert _safe_eval("sqrt(64 + 225)") == "17"


def test_safe_eval_combinatorics():
    assert _safe_eval("comb(5, 2)") == "10"
    assert _safe_eval("factorial(5)") == "120"


def test_safe_eval_rejects_invalid():
    assert "Error" in _safe_eval("import os")
    assert "Error" in _safe_eval("__import__('os')")
    assert "Error" in _safe_eval("(1).bit_length()")


# -- Answer extraction tests ---------------------------------------------------


def test_extract_boxed():
    assert _extract_boxed(r"Therefore \boxed{42}") == "42"
    assert _extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert _extract_boxed("no boxed here") is None


def test_extract_answer_boxed():
    assert _extract_answer(r"Therefore \boxed{42}") == "42"


def test_extract_answer_natural():
    assert _extract_answer("The final answer is 42.") == "42"


# -- Evaluator tests -----------------------------------------------------------
#
# The new convention: evaluator reads from
# episode.trajectories[-1].steps[-1].model_response. No artifacts mirroring.


def _episode_with_last_response(text: str) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="langgraph-math", steps=[Step(model_response=text)])],
    )


def test_evaluator_correct():
    task = {"answer": "10"}
    episode = _episode_with_last_response(r"After computing, \boxed{10}")
    result = langgraph_math_evaluator.evaluate(task, episode)
    assert result.is_correct is True
    assert result.reward == 1.0


def test_evaluator_wrong():
    task = {"answer": "10"}
    episode = _episode_with_last_response(r"\boxed{9}")
    result = langgraph_math_evaluator.evaluate(task, episode)
    assert result.is_correct is False


def test_evaluator_no_response():
    task = {"answer": "10"}
    episode = Episode(trajectories=[Trajectory(name="langgraph-math", steps=[])])
    result = langgraph_math_evaluator.evaluate(task, episode)
    assert result.is_correct is False


def test_evaluator_walks_back_past_empty_steps():
    """Tool-result steps may have empty model_response; walk back to the assistant turn."""
    task = {"answer": "10"}
    episode = Episode(
        trajectories=[
            Trajectory(
                name="langgraph-math",
                steps=[
                    Step(model_response="I'll calculate 5 + 5"),
                    Step(model_response=""),  # tool message turn
                    Step(model_response=r"The answer is \boxed{10}"),
                ],
            )
        ]
    )
    result = langgraph_math_evaluator.evaluate(task, episode)
    assert result.is_correct is True


def test_evaluator_latex_fraction():
    task = {"answer": r"\frac{1}{2}"}
    episode = _episode_with_last_response(r"\boxed{0.5}")
    result = langgraph_math_evaluator.evaluate(task, episode)
    assert result.is_correct is True
