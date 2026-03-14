"""Tests for EvalRunner two-stage pipeline."""

from __future__ import annotations

import asyncio

import pytest

from rllm.data.dataset import Dataset
from rllm.experimental.eval.results import EvalItem, EvalResult
from rllm.experimental.eval.runner import EvalRunner
from rllm.experimental.eval.types import AgentConfig, EvalOutput, Signal, Task
from rllm.types import Episode, Step, Trajectory

# ---------------------------------------------------------------------------
# Test agents and evaluators
# ---------------------------------------------------------------------------


class _PerfectAgent:
    def run(self, task: Task, config: AgentConfig) -> Episode:
        data = task.data if isinstance(task, Task) else task
        step = Step(input=data.get("question", ""), output="correct", done=True)
        return Episode(task=data, trajectories=[Trajectory(name="test", steps=[step])], artifacts={"answer": "correct"})


class _ErrorAgent:
    def run(self, task: Task, config: AgentConfig) -> Episode:
        raise RuntimeError("Simulated failure")


class _AlwaysCorrectEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(
            reward=1.0,
            is_correct=True,
            signals=[Signal(name="accuracy", value=1.0)],
        )


class _AlwaysWrongEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
        )


class _MultiSignalEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(
            reward=0.8,
            is_correct=True,
            signals=[
                Signal(name="accuracy", value=1.0),
                Signal(name="format", value=0.5),
            ],
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset():
    data = [{"question": f"q{i}", "ground_truth": f"a{i}"} for i in range(5)]
    return Dataset(data=data, name="test_ds", split="test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


def test_signals_on_items(small_dataset):
    runner = EvalRunner(base_url="http://fake", model="test")
    result = asyncio.run(runner.run(small_dataset, _PerfectAgent(), _MultiSignalEvaluator(), agent_name="multi"))

    for item in result.items:
        assert "accuracy" in item.signals
        assert "format" in item.signals
        assert item.signals["accuracy"] == 1.0
        assert item.signals["format"] == 0.5


def test_signal_averages(small_dataset):
    runner = EvalRunner(base_url="http://fake", model="test")
    result = asyncio.run(runner.run(small_dataset, _PerfectAgent(), _MultiSignalEvaluator(), agent_name="multi"))

    assert result.signal_averages["accuracy"] == pytest.approx(1.0)
    assert result.signal_averages["format"] == pytest.approx(0.5)


def test_reward_written_back_to_trajectories():
    """Verify that the runner writes reward and signals back onto trajectories."""
    data = [{"question": "q1", "ground_truth": "a1"}]
    dataset = Dataset(data=data, name="test_ds", split="test")

    # We need to capture the episode after the runner processes it.
    # The simplest way: use a custom agent that stores its episode.
    episodes: list[Episode] = []

    class _CapturingAgent:
        def run(self, task: Task, config: AgentConfig) -> Episode:
            data = task.data if isinstance(task, Task) else task
            step = Step(input="q", output="a", done=True)
            ep = Episode(task=data, trajectories=[Trajectory(name="t", steps=[step])])
            episodes.append(ep)
            return ep

    runner = EvalRunner(base_url="http://fake", model="test")
    asyncio.run(runner.run(dataset, _CapturingAgent(), _AlwaysCorrectEvaluator()))

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.is_correct is True
    assert ep.trajectories[0].reward == 1.0
    assert ep.trajectories[0].signals == {"accuracy": 1.0}


def test_result_summary_with_signals():
    items = [
        EvalItem(idx=0, reward=1.0, is_correct=True, signals={"accuracy": 1.0}),
        EvalItem(idx=1, reward=0.0, is_correct=False, signals={"accuracy": 0.0}),
    ]
    result = EvalResult.from_items("test", "model", "agent", items)
    assert result.signal_averages["accuracy"] == pytest.approx(0.5)

    summary = result.summary_table()
    assert "50.0%" in summary
