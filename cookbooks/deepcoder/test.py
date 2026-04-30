"""Tests for the deepcoder cookbook (no LLM, no real test execution)."""

from __future__ import annotations

from deepcoder_flow import (
    extract_code,
    format_feedback,
)
from evaluator import deepcoder_evaluator

from rllm.types import Episode, Step, Trajectory

# -- Code extraction -----------------------------------------------------------


def test_extract_code_python_block():
    text = "Here:\n```python\ndef f():\n    return 42\n```"
    assert extract_code(text) == "def f():\n    return 42"


def test_extract_code_py_alias():
    text = "```py\nprint('hi')\n```"
    assert extract_code(text) == "print('hi')"


def test_extract_code_takes_last_block():
    text = "First attempt:\n```python\nbad = 1\n```\nWait, let me revise:\n```python\ngood = 2\n```"
    assert extract_code(text) == "good = 2"


def test_extract_code_no_lang_tag():
    text = "```\nx = 1\n```"
    assert extract_code(text) == "x = 1"


def test_extract_code_none():
    assert extract_code("no code here") is None
    assert extract_code("`inline only`") is None


# -- Feedback rendering --------------------------------------------------------


def test_format_feedback_with_failures():
    meta = {
        "test_results": [
            {"passed": True, "input": "1", "expected": "1", "output": "1"},
            {"passed": False, "input": "2 3", "expected": "5", "output": "6"},
        ]
    }
    msg = format_feedback(meta)
    assert "Some test cases failed" in msg
    assert "Input: 2 3" in msg
    assert "Expected: 5" in msg
    assert "Got: 6" in msg
    assert "revised complete solution" in msg


def test_format_feedback_caps_at_max():
    meta = {"test_results": [{"passed": False, "input": str(i), "expected": "x", "output": "y"} for i in range(5)]}
    msg = format_feedback(meta, max_failures=2)
    # Only the first 2 failures should appear.
    assert "Input: 0" in msg
    assert "Input: 1" in msg
    assert "Input: 3" not in msg


def test_format_feedback_truncates_long_strings():
    long_input = "x" * 1000
    meta = {"test_results": [{"passed": False, "input": long_input, "expected": "z", "output": "y"}]}
    msg = format_feedback(meta)
    assert "(truncated)" in msg
    assert long_input not in msg


def test_format_feedback_no_failures():
    """When test_results is empty (e.g. infrastructure error), prompt for edge-case scrutiny."""
    msg = format_feedback({})
    assert "edge cases" in msg


# -- Evaluator -----------------------------------------------------------------


def _ep(answer: str, passed: bool = False) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="deepcoder", steps=[Step()])],
        artifacts={"answer": answer, "passed": passed, "turns": 1},
        is_correct=passed,
    )


def test_evaluator_returns_evaloutput():
    """Evaluator should always return EvalOutput regardless of whether the code passes."""
    task = {
        "question": "Add two numbers",
        "ground_truth": '[{"input": "1 2\\n", "output": "3", "testtype": "stdin_stdout", "metadata": {"func_name": null}}]',
        "data_source": "livecodebench",
    }
    out = deepcoder_evaluator.evaluate(task, _ep("```python\nbroken syntax\n```"))
    # Whatever the verdict, the output must conform to the protocol.
    assert hasattr(out, "reward")
    assert hasattr(out, "is_correct")
    assert isinstance(out.reward, float)
