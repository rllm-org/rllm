"""Tests for the deepcoder cookbook (no LLM, no real test execution)."""

from __future__ import annotations

import json

from deepcoder_eval import deepcoder_evaluator

from rllm.types import Episode, Step, Trajectory


def _ep(answer: str) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="deepcoder", steps=[Step()])],
        artifacts={"answer": answer},
    )


# -- Evaluator end-to-end ------------------------------------------------------


def test_evaluator_correct_solution():
    """A correct solution wrapped in ```python``` fences should score 1.0."""
    task = {
        "data_source": "livecodebench",
        "ground_truth": json.dumps(
            [
                {
                    "input": "1 2\n",
                    "output": "1 2",
                    "testtype": "stdin_stdout",
                    "metadata": {"func_name": None},
                }
            ]
        ),
    }
    answer = "Here is my solution:\n\n```python\nx, y = map(int, input().split())\nprint(min(x, y), max(x, y))\n```"
    out = deepcoder_evaluator.evaluate(task, _ep(answer))
    assert out.is_correct is True
    assert out.reward == 1.0


def test_evaluator_wrong_solution():
    """A wrong-output solution should score 0.0 (and not crash)."""
    task = {
        "data_source": "livecodebench",
        "ground_truth": json.dumps([{"input": "1\n", "output": "42", "testtype": "stdin_stdout", "metadata": {"func_name": None}}]),
    }
    answer = "```python\nprint('nope')\n```"
    out = deepcoder_evaluator.evaluate(task, _ep(answer))
    assert out.is_correct is False
    assert out.reward == 0.0


def test_evaluator_no_code_block():
    """Reply without a fenced code block scores 0 (format error)."""
    task = {
        "data_source": "livecodebench",
        "ground_truth": json.dumps([{"input": "1\n", "output": "1", "testtype": "stdin_stdout", "metadata": {"func_name": None}}]),
    }
    out = deepcoder_evaluator.evaluate(task, _ep("I would solve it like this..."))
    assert out.is_correct is False
    assert out.reward == 0.0


def test_evaluator_picks_last_code_block():
    """When the model emits multiple code blocks, only the last one is graded."""
    task = {
        "data_source": "livecodebench",
        "ground_truth": json.dumps(
            [
                {
                    "input": "3 4\n",
                    "output": "7",
                    "testtype": "stdin_stdout",
                    "metadata": {"func_name": None},
                }
            ]
        ),
    }
    answer = "First attempt:\n```python\nprint('wrong')\n```\nActually let me revise:\n```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    out = deepcoder_evaluator.evaluate(task, _ep(answer))
    assert out.is_correct is True


def test_evaluator_handles_task_object():
    """Evaluator accepts a rllm.types.Task too, not just a dict."""
    from pathlib import Path

    from rllm.types import Task

    task_meta = {
        "data_source": "livecodebench",
        "ground_truth": json.dumps([{"input": "5\n", "output": "5", "testtype": "stdin_stdout", "metadata": {"func_name": None}}]),
    }
    task = Task(id="t1", instruction="", metadata=task_meta, dataset_dir=Path("."))
    answer = "```python\nprint(input().strip())\n```"
    out = deepcoder_evaluator.evaluate(task, _ep(answer))
    assert out.is_correct is True
