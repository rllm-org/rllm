import asyncio

from rllm.agents.agent import Episode, Trajectory
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
    def __init__(self):
        self.created = None
        self.deleted = None

    async def acreate_session(self, session_id, is_validation=False):
        self.created = (session_id, is_validation)

    def get_session_url(self, session_id):
        return f"http://gateway/{session_id}"

    async def aget_traces(self, session_id):
        return []

    async def adelete_session(self, session_id):
        self.deleted = session_id


def test_run_single_passes_validation_flag_and_preserves_termination_reason():
    agent = _Agent()
    gateway = _Gateway()
    engine = AgentFlowEngine(
        agent_flow=agent,
        evaluator=_Evaluator(),
        gateway=gateway,
        model="test-model",
        n_parallel_tasks=1,
    )

    try:
        episode = asyncio.run(engine._run_single({"question": "q"}, "task:0", is_validation=True))
    finally:
        engine.shutdown()

    assert gateway.created == ("task:0", True)
    assert gateway.deleted == "task:0"
    assert agent.config.is_validation is True
    assert agent.config.session_uid == "task:0"
    assert episode.termination_reason == TerminationReason.ERROR
