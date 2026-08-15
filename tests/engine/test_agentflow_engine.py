import asyncio

import pytest

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.data.utils import task_from_row
from rllm.engine.agentflow_engine import AgentFlowEngine, EnrichMismatchError, enrich_episode_with_traces
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


def _trace(
    session_id: str,
    index: int,
    *,
    prompt_token_ids: list[int] | None = None,
    completion_token_ids: list[int] | None = None,
):
    from rllm_model_gateway.models import TraceRecord

    prompt_ids = [index + 1] if prompt_token_ids is None else prompt_token_ids
    completion_ids = [index + 11] if completion_token_ids is None else completion_token_ids
    return TraceRecord(
        trace_id=f"trace-{index}",
        session_id=session_id,
        model="m",
        messages=[{"role": "user", "content": "Q"}],
        response_message={"role": "assistant", "content": f"A{index}"},
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        logprobs=[-0.1] * len(completion_ids),
        finish_reason="stop",
        metadata={},
    )


def _empty_token_trace(session_id: str):
    return _trace(session_id, 0, prompt_token_ids=[], completion_token_ids=[])


@pytest.mark.parametrize("strict", [True, False])
def test_enrichment_rejects_extra_valid_trace_when_all_trajectories_have_steps(strict):
    episode = Episode(trajectories=[Trajectory(name="solver", steps=[Step()])])

    with pytest.raises(EnrichMismatchError, match=r"traces=2 agent_steps=1"):
        enrich_episode_with_traces(
            episode,
            [_trace("session", 0), _trace("session", 1)],
            "session",
            {},
            strict=strict,
        )


def test_enrichment_drops_trailing_malformed_trace():
    episode = Episode(trajectories=[Trajectory(name="solver", steps=[Step()])])

    enriched = enrich_episode_with_traces(
        episode,
        [
            _trace("session", 0),
            _trace("session", 1, prompt_token_ids=[], completion_token_ids=[]),
        ],
        "session",
        {},
    )

    assert [step.id for step in enriched.trajectories[0].steps] == ["trace-0"]


def test_enrichment_assigns_extra_traces_to_trajectory_without_steps():
    episode = Episode(
        trajectories=[
            Trajectory(name="reported", steps=[Step()]),
            Trajectory(name="trace-driven"),
        ]
    )

    enriched = enrich_episode_with_traces(
        episode,
        [_trace("session", 0), _trace("session", 1), _trace("session", 2)],
        "session",
        {},
    )

    assert [step.id for step in enriched.trajectories[0].steps] == ["trace-0"]
    assert [step.id for step in enriched.trajectories[1].steps] == ["trace-1", "trace-2"]


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
            with pytest.raises(EnrichMismatchError):
                asyncio.run(engine._run_single(task, "task:0", is_validation=False))
    finally:
        engine.shutdown()


def test_needs_env_flow_must_declare_env_param():
    """Binding a needs_env flow whose run() lacks the keyword-only ``env``
    parameter fails at construction, not mid-rollout."""
    from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

    class _LegacyFlow(SandboxedAgentFlow):
        def run(self, task, config):  # no env param
            return None

    with pytest.raises(TypeError, match="keyword-only 'env'"):
        AgentFlowEngine(
            agent_flow=_LegacyFlow(),
            evaluator=_Evaluator(),
            gateway=_Gateway(),
            model="test-model",
            n_parallel_tasks=1,
        )


def test_env_flow_receives_sandbox_and_container_url():
    """A needs_env flow gets the hook-provisioned sandbox as ``env`` and, when
    its LLM client runs in-sandbox on docker, a container-reachable URL."""
    from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

    seen = {}

    class _EnvFlow(SandboxedAgentFlow):
        llm_inside_env = True

        def run(self, task, config, *, env):
            seen["env"] = env
            seen["base_url"] = config.base_url
            return None

    sandbox = object()

    class _Hooks:
        def setup(self, task, agent_flow, uid):
            from rllm.engine.agentflow_engine import TaskContext

            return TaskContext(evaluator=_Evaluator(), env=sandbox, env_backend="docker")

    class _LoopbackGateway(_Gateway):
        def get_session_url(self, session_id, public=True):
            return f"http://127.0.0.1:9131/sessions/{session_id}/v1"

    engine = AgentFlowEngine(
        agent_flow=_EnvFlow(),
        evaluator=None,
        gateway=_LoopbackGateway(),
        model="test-model",
        n_parallel_tasks=1,
        hooks=_Hooks(),
    )
    task = task_from_row({"question": "q"}, "task")
    try:
        asyncio.run(engine._run_single(task, "task:0", is_validation=True))
    finally:
        engine.shutdown()

    assert seen["env"] is sandbox
    assert seen["base_url"].startswith("http://host.docker.internal:9131/")
