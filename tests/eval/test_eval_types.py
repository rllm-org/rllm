"""Tests for eval types: protocols, AgentFlow instances, and evaluators."""

from __future__ import annotations

import pytest

from rllm.experimental.eval.types import (
    AgentConfig,
    AgentFlow,
    CodeEvaluator,
    CompoundEvaluator,
    CountdownEvaluator,
    EvalOutput,
    Evaluator,
    F1Evaluator,
    MathEvaluator,
    Signal,
    Task,
    _extract_agent_answer,
)
from rllm.types import Episode, Step, Trajectory

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class _DummyAgent:
    def run(self, task: Task, config: AgentConfig) -> Episode:
        return Episode(task=task.data, trajectories=[], artifacts={"answer": "42"})


class _DummyEvaluator:
    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True)


def test_agent_flow_protocol():
    agent = _DummyAgent()
    assert isinstance(agent, AgentFlow)


def test_evaluator_protocol():
    evaluator = _DummyEvaluator()
    assert isinstance(evaluator, Evaluator)


def test_react_agent_is_agent_flow():
    from rllm.experimental.agents import react_agent

    assert isinstance(react_agent, AgentFlow), "react_agent is not an AgentFlow"


def test_builtin_evaluators_are_evaluators():
    for cls in [MathEvaluator, CountdownEvaluator, CodeEvaluator, F1Evaluator]:
        evaluator = cls()
        assert isinstance(evaluator, Evaluator), f"{cls.__name__} is not an Evaluator"


# ---------------------------------------------------------------------------
# _extract_agent_answer
# ---------------------------------------------------------------------------


def test_extract_answer_from_artifacts():
    ep = Episode(artifacts={"answer": "42"})
    assert _extract_agent_answer(ep) == "42"


def test_extract_answer_from_trajectory_output():
    traj = Trajectory(name="t", output="hello")
    ep = Episode(trajectories=[traj])
    assert _extract_agent_answer(ep) == "hello"


def test_extract_answer_from_last_step():
    step = Step(output="step_answer")
    traj = Trajectory(name="t", steps=[step])
    ep = Episode(trajectories=[traj])
    assert _extract_agent_answer(ep) == "step_answer"


def test_extract_answer_empty_episode():
    ep = Episode()
    assert _extract_agent_answer(ep) == ""


# ---------------------------------------------------------------------------
# MathEvaluator
# ---------------------------------------------------------------------------


class TestMathEvaluator:
    def test_correct_answer(self):
        evaluator = MathEvaluator()
        task = {"ground_truth": "4", "data_source": "test"}
        ep = Episode(artifacts={"answer": "The answer is \\boxed{4}"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_answer(self):
        evaluator = MathEvaluator()
        task = {"ground_truth": "4", "data_source": "test"}
        ep = Episode(artifacts={"answer": "The answer is \\boxed{5}"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_no_boxed_answer(self):
        evaluator = MathEvaluator()
        task = {"ground_truth": "4", "data_source": "test"}
        ep = Episode(artifacts={"answer": "The answer is 4"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False
        assert result.reward == 0.0

    def test_no_ground_truth(self):
        evaluator = MathEvaluator()
        task = {"data_source": "test"}
        ep = Episode(artifacts={"answer": "\\boxed{4}"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_signals_present(self):
        evaluator = MathEvaluator()
        task = {"ground_truth": "4", "data_source": "test"}
        ep = Episode(artifacts={"answer": "\\boxed{4}"})
        result = evaluator.evaluate(task, ep)
        assert len(result.signals) > 0
        assert result.signals[0].name == "accuracy"


# ---------------------------------------------------------------------------
# CountdownEvaluator
# ---------------------------------------------------------------------------


class TestCountdownEvaluator:
    def test_correct_countdown(self):
        evaluator = CountdownEvaluator()
        task = {"target": 10, "nums": [2, 3, 5]}
        ep = Episode(artifacts={"answer": "<answer>2 + 3 + 5</answer>"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is True
        assert result.reward == 1.0

    def test_wrong_countdown(self):
        evaluator = CountdownEvaluator()
        task = {"target": 10, "nums": [2, 3, 5]}
        ep = Episode(artifacts={"answer": "<answer>2 + 3</answer>"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False

    def test_missing_target(self):
        evaluator = CountdownEvaluator()
        task = {"nums": [2, 3, 5]}
        ep = Episode(artifacts={"answer": "<answer>2 + 3 + 5</answer>"})
        result = evaluator.evaluate(task, ep)
        assert result.is_correct is False


# ---------------------------------------------------------------------------
# F1Evaluator
# ---------------------------------------------------------------------------


class TestF1Evaluator:
    def test_exact_match(self):
        evaluator = F1Evaluator()
        task = {"ground_truth": "Shakespeare"}
        ep = Episode(artifacts={"answer": "Shakespeare"})
        result = evaluator.evaluate(task, ep)
        assert result.reward == pytest.approx(1.0)
        assert result.is_correct is True

    def test_partial_match(self):
        evaluator = F1Evaluator()
        task = {"ground_truth": "William Shakespeare"}
        ep = Episode(artifacts={"answer": "Shakespeare wrote Hamlet"})
        result = evaluator.evaluate(task, ep)
        assert 0.0 < result.reward < 1.0
        assert result.is_correct is True

    def test_no_match(self):
        evaluator = F1Evaluator()
        task = {"ground_truth": "Shakespeare"}
        ep = Episode(artifacts={"answer": "Dickens"})
        result = evaluator.evaluate(task, ep)
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_empty_prediction(self):
        evaluator = F1Evaluator()
        task = {"ground_truth": "Shakespeare"}
        ep = Episode(artifacts={"answer": ""})
        result = evaluator.evaluate(task, ep)
        assert result.reward == 0.0

    def test_signals(self):
        evaluator = F1Evaluator()
        task = {"ground_truth": "hello world"}
        ep = Episode(artifacts={"answer": "hello world"})
        result = evaluator.evaluate(task, ep)
        assert any(s.name == "f1" for s in result.signals)


# ---------------------------------------------------------------------------
# CompoundEvaluator
# ---------------------------------------------------------------------------


class TestCompoundEvaluator:
    def test_weighted_average(self):
        e1 = _DummyEvaluator()  # reward=1.0
        e2 = _FixedEvaluator(0.5)
        compound = CompoundEvaluator([(e1, 1.0), (e2, 1.0)])
        task = {}
        ep = Episode()
        result = compound.evaluate(task, ep)
        assert result.reward == pytest.approx(0.75)

    def test_any_correct(self):
        e1 = _FixedEvaluator(0.0, is_correct=False)
        e2 = _FixedEvaluator(1.0, is_correct=True)
        compound = CompoundEvaluator([(e1, 1.0), (e2, 1.0)])
        result = compound.evaluate({}, Episode())
        assert result.is_correct is True


class _FixedEvaluator:
    def __init__(self, reward: float, is_correct: bool | None = None):
        self._reward = reward
        self._is_correct = is_correct if is_correct is not None else (reward > 0)

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(
            reward=self._reward,
            is_correct=self._is_correct,
            signals=[Signal(name="test", value=self._reward)],
        )


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


def test_agent_config_defaults():
    config = AgentConfig(base_url="http://localhost:8000", model="test-model", session_uid="s1")
    assert config.metadata == {}


# ---------------------------------------------------------------------------
# Signal and EvalOutput
# ---------------------------------------------------------------------------


def test_signal_creation():
    sig = Signal(name="accuracy", value=0.95)
    assert sig.name == "accuracy"
    assert sig.value == 0.95
    assert sig.metadata == {}


def test_eval_output_creation():
    output = EvalOutput(
        reward=1.0,
        is_correct=True,
        signals=[Signal(name="accuracy", value=1.0)],
    )
    assert output.reward == 1.0
    assert len(output.signals) == 1


# ---------------------------------------------------------------------------
# Trajectory.signals and Episode.artifacts
# ---------------------------------------------------------------------------


def test_trajectory_signals_field():
    traj = Trajectory(name="t")
    assert traj.signals == {}
    traj.signals = {"accuracy": 1.0, "f1": 0.8}
    assert traj.signals["accuracy"] == 1.0


def test_episode_artifacts_field():
    ep = Episode(artifacts={"answer": "42", "code": "print(42)"})
    assert ep.artifacts["answer"] == "42"
    assert ep.artifacts["code"] == "print(42)"


def test_episode_id_auto_generated():
    ep1 = Episode()
    ep2 = Episode()
    assert ep1.id != ""
    assert ep2.id != ""
    assert ep1.id != ep2.id


def test_episode_id_can_be_set():
    ep = Episode(id="custom-id")
    assert ep.id == "custom-id"
