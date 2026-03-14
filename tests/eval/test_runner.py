"""Tests for EvalRunner (basic contract tests)."""

from __future__ import annotations

import asyncio

import pytest

from rllm.data.dataset import Dataset
from rllm.experimental.eval.results import EvalItem, EvalResult
from rllm.experimental.eval.runner import EvalRunner
from rllm.experimental.eval.types import AgentConfig, EvalOutput, Signal, Task
from rllm.types import Episode, Step, Trajectory


class _PerfectAgent:
    """Agent that always returns a trajectory."""

    def run(self, task: Task, config: AgentConfig) -> Episode:
        data = task.data if isinstance(task, Task) else task
        step = Step(input=data.get("question", ""), output="correct", reward=1.0, done=True)
        return Episode(task=data, trajectories=[Trajectory(name="test", steps=[step])], artifacts={"answer": "correct"})


class _ErrorAgent:
    """Agent that always raises an exception."""

    def run(self, task: Task, config: AgentConfig) -> Episode:
        raise RuntimeError("Simulated failure")


class _AlwaysCorrectEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True, signals=[Signal(name="accuracy", value=1.0)])


class _AlwaysWrongEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=0.0, is_correct=False, signals=[Signal(name="accuracy", value=0.0)])


@pytest.fixture
def small_dataset():
    data = [{"question": f"q{i}", "ground_truth": f"a{i}"} for i in range(5)]
    return Dataset(data=data, name="test_ds", split="test")


def test_perfect_score(small_dataset):
    runner = EvalRunner(base_url="http://fake", model="test")
    result = asyncio.run(runner.run(small_dataset, _PerfectAgent(), _AlwaysCorrectEvaluator(), agent_name="perfect"))

    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    assert result.correct == 5
    assert result.total == 5
    assert result.errors == 0


def test_zero_score(small_dataset):
    runner = EvalRunner(base_url="http://fake", model="test")
    result = asyncio.run(runner.run(small_dataset, _PerfectAgent(), _AlwaysWrongEvaluator(), agent_name="failing"))

    assert result.score == 0.0
    assert result.correct == 0
    assert result.total == 5


def test_error_handling(small_dataset):
    runner = EvalRunner(base_url="http://fake", model="test")
    result = asyncio.run(runner.run(small_dataset, _ErrorAgent(), _AlwaysCorrectEvaluator(), agent_name="error"))

    assert result.errors == 5
    assert result.score == 0.0
    assert all(item.error is not None for item in result.items)


def test_result_summary():
    items = [
        EvalItem(idx=0, reward=1.0, is_correct=True),
        EvalItem(idx=1, reward=0.0, is_correct=False),
        EvalItem(idx=2, reward=1.0, is_correct=True),
    ]
    result = EvalResult.from_items("test", "model", "agent", items)

    assert result.score == pytest.approx(2 / 3)
    assert result.correct == 2
    assert result.total == 3

    summary = result.summary_table()
    assert "66.7%" in summary
    assert "2/3" in summary


def test_result_save(tmp_path):
    items = [EvalItem(idx=0, reward=1.0, is_correct=True)]
    result = EvalResult.from_items("test", "model", "agent", items)

    path = str(tmp_path / "results.json")
    saved_path = result.save(path)
    assert saved_path == path
    assert (tmp_path / "results.json").exists()

    import json

    with open(path) as f:
        data = json.load(f)
    assert data["score"] == 1.0
    assert data["total"] == 1
