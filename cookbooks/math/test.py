"""Tests for the math cookbook (no LLM)."""

from __future__ import annotations

from math_eval import math_evaluator

from rllm.types import Episode, Step, Trajectory


def _ep(answer: str) -> Episode:
    return Episode(
        trajectories=[Trajectory(name="math", steps=[Step()])],
        artifacts={"answer": answer},
    )


def test_evaluator_correct_boxed_answer():
    task = {"ground_truth": "42"}
    out = math_evaluator.evaluate(task, _ep("After simplifying, \\boxed{42}."))
    assert out.is_correct is True
    assert out.reward == 1.0


def test_evaluator_wrong_boxed_answer():
    task = {"ground_truth": "42"}
    out = math_evaluator.evaluate(task, _ep("My answer is \\boxed{41}."))
    assert out.is_correct is False
    assert out.reward == 0.0


def test_evaluator_no_boxed_answer():
    """Reply without \\boxed{} returns reward=0 with reason=no_answer_extracted."""
    task = {"ground_truth": "42"}
    out = math_evaluator.evaluate(task, _ep("The answer is 42."))
    assert out.is_correct is False
    assert out.reward == 0.0


def test_evaluator_takes_last_boxed():
    task = {"ground_truth": "5"}
    answer = "First I thought \\boxed{4}, but actually \\boxed{5}."
    out = math_evaluator.evaluate(task, _ep(answer))
    assert out.is_correct is True


def test_evaluator_latex_fraction_equivalence():
    """Symbolic grading: 0.5 should match \\frac{1}{2}."""
    task = {"ground_truth": "\\frac{1}{2}"}
    out = math_evaluator.evaluate(task, _ep("\\boxed{0.5}"))
    assert out.is_correct is True


def test_evaluator_handles_task_object():
    """Eval-Runner path: task is a rllm.types.Task, not a dict."""
    from pathlib import Path

    from rllm.types import Task

    task = Task(id="t1", instruction="", metadata={"ground_truth": "7"}, dataset_dir=Path("."))
    out = math_evaluator.evaluate(task, _ep("\\boxed{7}"))
    assert out.is_correct is True


def test_evaluator_no_ground_truth_returns_zero():
    """Missing ground_truth shouldn't crash, just score 0."""
    task: dict = {}
    out = math_evaluator.evaluate(task, _ep("\\boxed{99}"))
    assert out.is_correct is False
    assert out.reward == 0.0
