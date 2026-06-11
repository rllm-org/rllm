import asyncio

import pytest

from rllm.agents.agent import Episode, Trajectory
from rllm.data.utils import task_from_row
from rllm.engine.agentflow_engine import AgentFlowEngine
from rllm.eval.types import EvalOutput
from rllm.workflows.workflow import TerminationReason


class _Agent:
    def __init__(self):
        self.config = None

    async def arun(self, task, config):
        self.config = config
        return Episode(
            id=task.id,
            termination_reason=TerminationReason.ERROR,
            trajectories=[Trajectory(name="solver")],
        )


class _Evaluator:
    def evaluate(self, task, episode):
        return EvalOutput(reward=0.0, is_correct=False)


class _Gateway:
    """Minimal gateway double; stocked traces are returned for every session."""

    def __init__(self, traces=None):
        self.created = None
        self.deleted = None
        self._traces = traces or []

    async def acreate_session(self, session_id, is_validation=False, sampling_params=None):
        self.created = (session_id, is_validation)

    def get_session_url(self, session_id, public=True):
        return f"http://gateway/{session_id}"

    async def aget_traces(self, session_id):
        return self._traces

    async def adelete_session(self, session_id):
        self.deleted = session_id

    async def adelete_sessions(self, session_ids):
        self.deleted = session_ids[-1] if session_ids else None


def test_run_single_passes_validation_flag_and_preserves_termination_reason():
    agent = _Agent()
    gateway = _Gateway()
    engine = AgentFlowEngine(
        agent_flow=agent,
        evaluator=_Evaluator(),
        gateway=gateway,
        model="test-model",
        n_parallel_tasks=1,
        val_sampling_params={"temperature": 0.1},
    )
    task = task_from_row({"question": "q"}, "task")

    try:
        episode = asyncio.run(engine._run_single(task, "task:0", is_validation=True))
    finally:
        engine.shutdown()

    assert gateway.created == ("task:0", True)
    assert agent.config.is_validation is True
    assert agent.config.session_uid == "task:0"
    assert episode.termination_reason == TerminationReason.ERROR


def _empty_token_trace(session_id: str):
    from rllm_model_gateway.models import TraceRecord

    return TraceRecord(
        trace_id=f"t-{session_id}",
        session_id=session_id,
        model="m",
        messages=[{"role": "user", "content": "Q"}],
        response_message={"role": "assistant", "content": "A"},
        prompt_token_ids=[],  # empty → corrupts loss math if it reaches training
        completion_token_ids=[],
        logprobs=[],
        finish_reason="stop",
        metadata={},
    )


@pytest.mark.parametrize("is_validation", [False, True])
def test_strict_enrichment_follows_is_validation(is_validation):
    """Training rollouts must reject empty token IDs (EnrichMismatchError →
    retry); validation tolerates them (evaluators read message text). The old
    ``strict = hooks is None`` proxy silently disabled this for sandboxed
    training, which always has hooks."""

    @__import__("rllm").rollout(name="noop")
    def noop_flow(task, config):
        return None

    gateway = _Gateway(traces=[_empty_token_trace("task:0")])
    engine = AgentFlowEngine(
        agent_flow=noop_flow,
        evaluator=_Evaluator(),
        gateway=gateway,
        model="test-model",
        n_parallel_tasks=1,
        retry_limit=1,
    )
    task = task_from_row({"question": "q"}, "task")

    try:
        if is_validation:
            episode = asyncio.run(engine._run_single(task, "task:0", is_validation=True))
            assert episode is not None
        else:
            from rllm.engine.agentflow_engine import EnrichMismatchError

            with pytest.raises(EnrichMismatchError):
                asyncio.run(engine._run_single(task, "task:0", is_validation=False))
    finally:
        engine.shutdown()
