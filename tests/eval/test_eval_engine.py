"""End-to-end test for ``AgentFlowEngine`` in eval mode (with EvalHooks).

The engine drives the same loop for training and eval — the difference is
whether ``hooks`` are installed. This test verifies the eval path works:

1. A flow that ``return None``s.
2. A fake gateway that captures whatever URL the flow's HTTP call would
   hit and returns canned ``TraceRecord`` objects on ``aget_traces``.
3. ``EvalHooks`` with an ``evaluator_override`` so we don't need a real
   verifier on disk.
4. Assert: the evaluator received an Episode with populated Steps (from
   the gateway traces), the reward was written back to the trajectory,
   and ``is_correct`` reflects the evaluator's verdict.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rllm_model_gateway.models import TraceRecord

import rllm
from rllm.engine.agentflow_engine import AgentFlowEngine
from rllm.eval._hooks import EvalHooks
from rllm.eval.types import EvalOutput
from rllm.types import AgentConfig, Episode, Task

# ---------------------------------------------------------------------------
# Fake flow + gateway + evaluator
# ---------------------------------------------------------------------------


@rllm.rollout(name="fake-flow")
async def fake_flow(task: Task, config: AgentConfig) -> None:
    """Return None — relies on the gateway to capture the trace."""
    return None


def _make_trace(content: str, *, session_id: str) -> TraceRecord:
    """Construct a minimal valid TraceRecord (no token IDs — eval doesn't need them)."""
    return TraceRecord(
        trace_id=f"t-{session_id}",
        session_id=session_id,
        model="fake-model",
        messages=[{"role": "user", "content": "Q"}],
        response_message={"role": "assistant", "content": content},
        prompt_token_ids=[],
        completion_token_ids=[],
        logprobs=[],
        finish_reason="stop",
        metadata={},
    )


class _FakeGateway:
    """In-memory stand-in for ``GatewayManager`` for unit tests.

    Implements only the async methods ``AgentFlowEngine`` uses. Each session
    is pre-stocked with a single canned response that the test specifies.
    """

    def __init__(self, response_per_uid: dict[str, str]):
        self._responses = response_per_uid
        self._sessions: set[str] = set()
        self.create_calls: list[str] = []
        self.delete_calls: list[str] = []

    async def acreate_session(self, session_id: str, is_validation: bool = False) -> str:
        self._sessions.add(session_id)
        self.create_calls.append(session_id)
        return session_id

    def get_session_url(self, session_id: str) -> str:
        return f"http://fake-gateway/sessions/{session_id}/v1"

    async def aget_traces(self, session_id: str) -> list[TraceRecord]:
        if session_id not in self._sessions:
            return []
        return [_make_trace(self._responses[session_id], session_id=session_id)]

    async def adelete_session(self, session_id: str) -> int:
        self._sessions.discard(session_id)
        self.delete_calls.append(session_id)
        return 1


class _StubEvaluator:
    """Reads the last assistant message from the enriched Episode.

    Mirrors what real eval-side evaluators do (e.g. cookbook
    ``math_evaluator``) — confirms the engine actually populates Steps
    from gateway traces before calling the evaluator.
    """

    def __init__(self, ground_truth: str):
        self._truth = ground_truth
        self.calls: list[tuple] = []

    def evaluate(self, task, episode: Episode) -> EvalOutput:
        self.calls.append((task, episode))
        if not episode.trajectories or not episode.trajectories[-1].steps:
            return EvalOutput(reward=0.0, is_correct=False)
        last = episode.trajectories[-1].steps[-1]
        is_correct = self._truth in (last.model_response or "")
        return EvalOutput(reward=1.0 if is_correct else 0.0, is_correct=is_correct)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_engine_populates_steps_from_gateway_traces():
    """The evaluator sees Steps populated from gateway traces, not the empty
    Trajectory the flow returned via ``return None``."""

    gateway = _FakeGateway(response_per_uid={"task-0:0": r"The answer is \boxed{42}"})
    evaluator = _StubEvaluator(ground_truth="42")
    hooks = EvalHooks(evaluator_override=evaluator)

    engine = AgentFlowEngine(
        agent_flow=fake_flow,
        evaluator=None,
        gateway=gateway,  # type: ignore[arg-type]
        model="fake-model",
        n_parallel_tasks=2,
        retry_limit=1,
        raise_on_error=True,
        hooks=hooks,
    )

    task = Task(id="task-0", instruction="What is the answer?", metadata={"answer": "42"}, dataset_dir=Path("."))

    try:
        episodes = asyncio.run(engine.execute_tasks([task], task_ids=["task-0"], is_validation=True))
    finally:
        engine.shutdown()

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep is not None
    assert ep.is_correct is True

    # Evaluator was called with an Episode whose trajectory has Steps from the gateway.
    assert len(evaluator.calls) == 1
    _, observed_episode = evaluator.calls[0]
    assert observed_episode.trajectories
    assert observed_episode.trajectories[-1].steps  # populated by enrichment
    assert observed_episode.trajectories[-1].steps[-1].model_response == r"The answer is \boxed{42}"

    # Reward was written back onto the trajectory.
    assert ep.trajectories[-1].reward == 1.0

    # Gateway lifecycle: session was created, then deleted.
    assert gateway.create_calls == ["task-0:0"]
    assert gateway.delete_calls == ["task-0:0"]


def test_eval_engine_marks_wrong_answer_incorrect():
    gateway = _FakeGateway(response_per_uid={"task-0:0": r"\boxed{99}"})
    evaluator = _StubEvaluator(ground_truth="42")
    hooks = EvalHooks(evaluator_override=evaluator)

    engine = AgentFlowEngine(
        agent_flow=fake_flow,
        evaluator=None,
        gateway=gateway,  # type: ignore[arg-type]
        model="fake-model",
        n_parallel_tasks=1,
        retry_limit=1,
        raise_on_error=True,
        hooks=hooks,
    )
    task = Task(id="task-0", instruction="?", metadata={"answer": "42"}, dataset_dir=Path("."))

    try:
        (ep,) = asyncio.run(engine.execute_tasks([task], task_ids=["task-0"], is_validation=True))
    finally:
        engine.shutdown()

    assert ep.is_correct is False
    assert ep.trajectories[-1].reward == 0.0


def test_eval_engine_runs_hook_teardown_on_success_and_failure():
    """Verify the hook's teardown closure runs whether the rollout succeeds or raises."""

    gateway = _FakeGateway(response_per_uid={"task-0:0": "ok"})
    evaluator = _StubEvaluator(ground_truth="ok")

    teardown_calls: list[str] = []

    class _RecordingHooks:
        def setup(self, task, agent_flow, uid):
            from rllm.engine.agentflow_engine import TaskContext

            teardown_calls.append(f"setup-{uid}")
            return TaskContext(
                evaluator=evaluator,
                teardown=lambda: teardown_calls.append(f"teardown-{uid}"),
            )

    engine = AgentFlowEngine(
        agent_flow=fake_flow,
        evaluator=None,
        gateway=gateway,  # type: ignore[arg-type]
        model="fake-model",
        n_parallel_tasks=1,
        retry_limit=1,
        raise_on_error=True,
        hooks=_RecordingHooks(),  # type: ignore[arg-type]
    )
    task = Task(id="task-0", instruction="?", metadata={}, dataset_dir=Path("."))

    try:
        asyncio.run(engine.execute_tasks([task], task_ids=["task-0"], is_validation=True))
    finally:
        engine.shutdown()

    # Setup ran, then teardown ran (success path)
    assert teardown_calls == ["setup-task-0:0", "teardown-task-0:0"]


def test_eval_engine_raises_when_neither_evaluator_nor_hooks_provided():
    gateway = MagicMock()
    with pytest.raises(ValueError, match="evaluator.*or.*hooks"):
        AgentFlowEngine(
            agent_flow=fake_flow,
            evaluator=None,
            gateway=gateway,
            model="fake",
            hooks=None,
        )
