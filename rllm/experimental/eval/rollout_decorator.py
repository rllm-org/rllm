"""Decorators that turn plain functions into AgentFlow / Evaluator objects.

``@rollout`` wraps a function so it satisfies the :class:`AgentFlow` protocol,
and ``@evaluator`` wraps a function so it satisfies the :class:`Evaluator`
protocol.  Both support bare (``@rollout``) and parameterized
(``@rollout(name="solver")``) syntax.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, overload

from rllm.experimental.eval.types import AgentConfig, EvalOutput, Task
from rllm.types import Episode, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return-value coercion helpers
# ---------------------------------------------------------------------------


def _coerce_to_episode(result: Any, task: Task, traj_name: str) -> Episode:
    """Convert a user function's return value into an Episode."""
    if isinstance(result, Episode):
        if result.task is None:
            result.task = task.data
        return result

    if isinstance(result, list) and result and isinstance(result[0], Trajectory):
        answer = ""
        if result and result[-1].output:
            answer = str(result[-1].output)
        return Episode(task=task.data, trajectories=result, artifacts={"answer": answer})

    if isinstance(result, str):
        traj = Trajectory(name=traj_name, steps=[])
        return Episode(task=task.data, trajectories=[traj], artifacts={"answer": result})

    if isinstance(result, dict):
        traj = Trajectory(name=traj_name, steps=[])
        answer = result.get("answer", "")
        return Episode(task=task.data, trajectories=[traj], artifacts={"answer": answer, **result})

    # Fallback: stringify
    traj = Trajectory(name=traj_name, steps=[])
    return Episode(task=task.data, trajectories=[traj], artifacts={"answer": str(result)})


def _coerce_to_eval_output(result: Any) -> EvalOutput:
    """Convert a user function's return value into an EvalOutput."""
    if isinstance(result, EvalOutput):
        return result

    if isinstance(result, bool):
        return EvalOutput(reward=1.0 if result else 0.0, is_correct=result)

    if isinstance(result, int | float):
        return EvalOutput(reward=float(result), is_correct=float(result) > 0)

    if isinstance(result, tuple) and len(result) == 2:
        reward, is_correct = result
        return EvalOutput(reward=float(reward), is_correct=bool(is_correct))

    raise TypeError(f"@evaluator function returned unsupported type {type(result).__name__}; expected EvalOutput, float, bool, or (float, bool)")


# ---------------------------------------------------------------------------
# AgentFlowFn — wrapper produced by @rollout
# ---------------------------------------------------------------------------


class AgentFlowFn:
    """AgentFlow wrapper that delegates to a plain function.

    Satisfies the :class:`AgentFlow` protocol (``run(task, config) -> Episode``).
    If the wrapped function is async, ``arun`` is also provided so that
    :func:`run_agent_flow` can await it directly.
    """

    def __init__(self, fn: Callable, *, name: str = "solver") -> None:
        self._fn = fn
        self._name = name
        self._is_async = inspect.iscoroutinefunction(fn)
        update_wrapper(self, fn)

    def run(self, task: Task, config: AgentConfig) -> Episode:
        if self._is_async:
            result = asyncio.run(self._fn(task, config))
        else:
            result = self._fn(task, config)
        return _coerce_to_episode(result, task, self._name)

    async def arun(self, task: Task, config: AgentConfig) -> Episode:
        if self._is_async:
            result = await self._fn(task, config)
        else:
            result = self._fn(task, config)
        return _coerce_to_episode(result, task, self._name)

    def __call__(self, task: Task, config: AgentConfig) -> Episode:
        return self.run(task, config)

    def __repr__(self) -> str:
        return f"AgentFlowFn({self._fn.__name__!r}, name={self._name!r})"


# ---------------------------------------------------------------------------
# EvaluatorFn — wrapper produced by @evaluator
# ---------------------------------------------------------------------------


class EvaluatorFn:
    """Evaluator wrapper that delegates to a plain function.

    Satisfies the :class:`Evaluator` protocol
    (``evaluate(task, episode) -> EvalOutput``).
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        update_wrapper(self, fn)

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        result = self._fn(task, episode)
        return _coerce_to_eval_output(result)

    def __call__(self, task: dict, episode: Episode) -> EvalOutput:
        return self.evaluate(task, episode)

    def __repr__(self) -> str:
        return f"EvaluatorFn({self._fn.__name__!r})"


# ---------------------------------------------------------------------------
# @rollout decorator
# ---------------------------------------------------------------------------


@overload
def rollout(fn: Callable) -> AgentFlowFn: ...


@overload
def rollout(*, name: str = "solver", register: str | None = None) -> Callable[[Callable], AgentFlowFn]: ...


def rollout(
    fn: Callable | None = None,
    *,
    name: str = "solver",
    register: str | None = None,
) -> AgentFlowFn | Callable[[Callable], AgentFlowFn]:
    """Decorator that turns a function into an :class:`AgentFlow`.

    The decorated function must accept ``(task, config)`` where *task* is a
    :class:`Task` and *config* is an :class:`AgentConfig`.  It may return:

    * a ``str`` — wrapped as the episode answer
    * an :class:`Episode` — passed through
    * a ``list[Trajectory]`` — wrapped in an Episode
    * a ``dict`` — treated as episode artifacts

    Examples::

        @rllm.rollout
        def solver(task, config):
            client = OpenAI(base_url=config.base_url, api_key="EMPTY")
            resp = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": task.data["question"]}],
            )
            return resp.choices[0].message.content

        @rllm.rollout(name="reasoning", register="my-agent")
        def reasoning_agent(task, config):
            ...

    Args:
        fn: The function to decorate (when used without parentheses).
        name: Trajectory name (default ``"solver"``).
        register: If provided, register the agent under this name in
            ``~/.rllm/agents.json`` for CLI discovery.
    """

    def _decorator(fn: Callable) -> AgentFlowFn:
        agent = AgentFlowFn(fn, name=name)
        if register is not None:
            from rllm.experimental.eval.agent_loader import register_agent

            register_agent(register, agent)
        return agent

    if fn is not None:
        return _decorator(fn)
    return _decorator


# ---------------------------------------------------------------------------
# @evaluator decorator
# ---------------------------------------------------------------------------


@overload
def evaluator(fn: Callable) -> EvaluatorFn: ...


@overload
def evaluator(*, register: str | None = None) -> Callable[[Callable], EvaluatorFn]: ...


def evaluator(
    fn: Callable | None = None,
    *,
    register: str | None = None,
) -> EvaluatorFn | Callable[[Callable], EvaluatorFn]:
    """Decorator that turns a function into an :class:`Evaluator`.

    The decorated function must accept ``(task, episode)`` where *task* is a
    ``dict`` and *episode* is an :class:`Episode`.  It may return:

    * an :class:`EvalOutput` — passed through
    * a ``float`` — reward value (``is_correct = reward > 0``)
    * a ``bool`` — correct/incorrect (reward 1.0 or 0.0)
    * a ``(float, bool)`` tuple — ``(reward, is_correct)``

    Examples::

        @rllm.evaluator
        def exact_match(task, episode):
            answer = episode.artifacts.get("answer", "")
            return 1.0 if answer.strip() == task["ground_truth"].strip() else 0.0

        @rllm.evaluator(register="my-eval")
        def custom_eval(task, episode):
            ...
            return EvalOutput(reward=score, is_correct=score > 0.5)

    Args:
        fn: The function to decorate (when used without parentheses).
        register: If provided, register the evaluator under this name in
            ``~/.rllm/evaluators.json`` for CLI discovery.
    """

    def _decorator(fn: Callable) -> EvaluatorFn:
        ev = EvaluatorFn(fn)
        if register is not None:
            from rllm.experimental.eval.evaluator_loader import register_evaluator

            register_evaluator(register, ev)
        return ev

    if fn is not None:
        return _decorator(fn)
    return _decorator
