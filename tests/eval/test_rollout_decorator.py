"""Tests for @rollout and @evaluator decorators."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from rllm.experimental.eval.rollout_decorator import (
    AgentFlowFn,
    EvaluatorFn,
    _coerce_to_episode,
    _coerce_to_eval_output,
    evaluator,
    rollout,
)
from rllm.experimental.eval.types import AgentConfig, AgentFlow, EvalOutput, Evaluator, Task, run_agent_flow
from rllm.types import Episode, Trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task():
    return Task(data={"question": "What is 2+2?", "ground_truth": "4"})


@pytest.fixture()
def config():
    return AgentConfig(base_url="http://localhost:4000/v1", model="test-model", session_uid="test-uid")


# ---------------------------------------------------------------------------
# @rollout tests
# ---------------------------------------------------------------------------


class TestRolloutBareDecorator:
    def test_creates_agent_flow_fn(self):
        @rollout
        def my_agent(task, config):
            return "hello"

        assert isinstance(my_agent, AgentFlowFn)

    def test_has_run_method(self):
        @rollout
        def my_agent(task, config):
            return "hello"

        assert hasattr(my_agent, "run")
        assert callable(my_agent.run)

    def test_has_arun_method(self):
        @rollout
        def my_agent(task, config):
            return "hello"

        assert hasattr(my_agent, "arun")

    def test_satisfies_agent_flow_protocol(self):
        @rollout
        def my_agent(task, config):
            return "hello"

        assert isinstance(my_agent, AgentFlow)

    def test_run_returns_episode(self, task, config):
        @rollout
        def my_agent(task, config):
            return "the answer is 4"

        episode = my_agent.run(task, config)
        assert isinstance(episode, Episode)
        assert episode.artifacts["answer"] == "the answer is 4"
        assert episode.task == task.data

    def test_default_trajectory_name(self, task, config):
        @rollout
        def my_agent(task, config):
            return "answer"

        episode = my_agent.run(task, config)
        assert len(episode.trajectories) == 1
        assert episode.trajectories[0].name == "solver"

    def test_callable(self, task, config):
        @rollout
        def my_agent(task, config):
            return "answer"

        episode = my_agent(task, config)
        assert isinstance(episode, Episode)


class TestRolloutParameterizedDecorator:
    def test_custom_name(self, task, config):
        @rollout(name="reasoning")
        def my_agent(task, config):
            return "answer"

        episode = my_agent.run(task, config)
        assert episode.trajectories[0].name == "reasoning"

    def test_register_calls_register_agent(self):
        with patch("rllm.experimental.eval.agent_loader.register_agent") as mock_reg:

            @rollout(register="my-agent")
            def my_agent(task, config):
                return "answer"

            mock_reg.assert_called_once_with("my-agent", my_agent)

    def test_repr(self):
        @rollout(name="solver")
        def my_agent(task, config):
            return "answer"

        assert "AgentFlowFn" in repr(my_agent)
        assert "my_agent" in repr(my_agent)


class TestRolloutReturnCoercion:
    def test_str_return(self, task, config):
        @rollout
        def my_agent(task, config):
            return "four"

        ep = my_agent.run(task, config)
        assert ep.artifacts["answer"] == "four"
        assert len(ep.trajectories) == 1
        assert ep.trajectories[0].steps == []

    def test_episode_return(self, task, config):
        @rollout
        def my_agent(task, config):
            return Episode(task=task.data, trajectories=[], artifacts={"answer": "direct"})

        ep = my_agent.run(task, config)
        assert ep.artifacts["answer"] == "direct"

    def test_episode_return_fills_task(self, task, config):
        @rollout
        def my_agent(task, config):
            return Episode(trajectories=[], artifacts={"answer": "x"})

        ep = my_agent.run(task, config)
        assert ep.task == task.data

    def test_trajectory_list_return(self, task, config):
        @rollout
        def my_agent(task, config):
            return [
                Trajectory(name="solver", steps=[], output="sol1"),
                Trajectory(name="judge", steps=[], output="sol2"),
            ]

        ep = my_agent.run(task, config)
        assert len(ep.trajectories) == 2
        assert ep.trajectories[0].name == "solver"
        assert ep.artifacts["answer"] == "sol2"

    def test_dict_return(self, task, config):
        @rollout
        def my_agent(task, config):
            return {"answer": "four", "confidence": 0.9}

        ep = my_agent.run(task, config)
        assert ep.artifacts["answer"] == "four"
        assert ep.artifacts["confidence"] == 0.9

    def test_fallback_to_str(self, task, config):
        @rollout
        def my_agent(task, config):
            return 42

        ep = my_agent.run(task, config)
        assert ep.artifacts["answer"] == "42"


class TestRolloutAsync:
    def test_async_function_via_run(self, task, config):
        @rollout
        async def my_agent(task, config):
            return "async answer"

        ep = my_agent.run(task, config)
        assert ep.artifacts["answer"] == "async answer"

    def test_async_function_via_arun(self, task, config):
        @rollout
        async def my_agent(task, config):
            return "async answer"

        ep = asyncio.run(my_agent.arun(task, config))
        assert ep.artifacts["answer"] == "async answer"

    def test_sync_function_via_arun(self, task, config):
        @rollout
        def my_agent(task, config):
            return "sync answer"

        ep = asyncio.run(my_agent.arun(task, config))
        assert ep.artifacts["answer"] == "sync answer"

    def test_works_with_run_agent_flow(self, task, config):
        @rollout
        def my_agent(task, config):
            return "hello"

        ep = asyncio.run(run_agent_flow(my_agent, task, config))
        assert isinstance(ep, Episode)
        assert ep.artifacts["answer"] == "hello"


# ---------------------------------------------------------------------------
# @evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluatorBareDecorator:
    def test_creates_evaluator_fn(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        assert isinstance(my_eval, EvaluatorFn)

    def test_has_evaluate_method(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        assert hasattr(my_eval, "evaluate")
        assert callable(my_eval.evaluate)

    def test_satisfies_evaluator_protocol(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        assert isinstance(my_eval, Evaluator)

    def test_evaluate_returns_eval_output(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        result = my_eval.evaluate({"ground_truth": "4"}, Episode(trajectories=[]))
        assert isinstance(result, EvalOutput)

    def test_callable(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        result = my_eval({"ground_truth": "4"}, Episode(trajectories=[]))
        assert isinstance(result, EvalOutput)


class TestEvaluatorParameterizedDecorator:
    def test_register_calls_register_evaluator(self):
        with patch("rllm.experimental.eval.evaluator_loader.register_evaluator") as mock_reg:

            @evaluator(register="my-eval")
            def my_eval(task, episode):
                return 1.0

            mock_reg.assert_called_once_with("my-eval", my_eval)

    def test_repr(self):
        @evaluator
        def my_eval(task, episode):
            return 1.0

        assert "EvaluatorFn" in repr(my_eval)
        assert "my_eval" in repr(my_eval)


class TestEvaluatorReturnCoercion:
    def test_eval_output_passthrough(self):
        @evaluator
        def my_eval(task, episode):
            return EvalOutput(reward=0.5, is_correct=False)

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 0.5
        assert result.is_correct is False

    def test_float_return(self):
        @evaluator
        def my_eval(task, episode):
            return 0.75

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 0.75
        assert result.is_correct is True

    def test_float_zero_is_not_correct(self):
        @evaluator
        def my_eval(task, episode):
            return 0.0

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_bool_true(self):
        @evaluator
        def my_eval(task, episode):
            return True

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 1.0
        assert result.is_correct is True

    def test_bool_false(self):
        @evaluator
        def my_eval(task, episode):
            return False

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_tuple_return(self):
        @evaluator
        def my_eval(task, episode):
            return (0.5, True)

        result = my_eval.evaluate({}, Episode(trajectories=[]))
        assert result.reward == 0.5
        assert result.is_correct is True

    def test_unsupported_type_raises(self):
        @evaluator
        def my_eval(task, episode):
            return "not valid"

        with pytest.raises(TypeError, match="unsupported type"):
            my_eval.evaluate({}, Episode(trajectories=[]))


# ---------------------------------------------------------------------------
# Coercion helper unit tests
# ---------------------------------------------------------------------------


class TestCoerceToEpisode:
    def test_episode_passthrough(self):
        task = Task(data={"q": "test"})
        ep = Episode(task={"q": "test"}, trajectories=[], artifacts={"answer": "x"})
        result = _coerce_to_episode(ep, task, "solver")
        assert result is ep

    def test_str(self):
        task = Task(data={"q": "test"})
        result = _coerce_to_episode("hello", task, "solver")
        assert result.artifacts["answer"] == "hello"

    def test_dict(self):
        task = Task(data={"q": "test"})
        result = _coerce_to_episode({"answer": "world", "extra": 1}, task, "solver")
        assert result.artifacts["answer"] == "world"
        assert result.artifacts["extra"] == 1


class TestCoerceToEvalOutput:
    def test_eval_output(self):
        eo = EvalOutput(reward=1.0, is_correct=True)
        assert _coerce_to_eval_output(eo) is eo

    def test_float(self):
        result = _coerce_to_eval_output(0.5)
        assert result.reward == 0.5
        assert result.is_correct is True

    def test_bool(self):
        result = _coerce_to_eval_output(False)
        assert result.reward == 0.0
        assert result.is_correct is False

    def test_tuple(self):
        result = _coerce_to_eval_output((0.3, False))
        assert result.reward == 0.3
        assert result.is_correct is False


# ---------------------------------------------------------------------------
# Top-level import test
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_rllm_rollout(self):
        import rllm

        assert rllm.rollout is rollout

    def test_rllm_evaluator(self):
        import rllm

        assert rllm.evaluator is evaluator
