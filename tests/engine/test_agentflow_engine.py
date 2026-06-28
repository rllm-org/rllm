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


def test_no_usable_model_output_detects_dead_upstream():
    """No LLM calls at all, or every call empty (no content, no tool_calls) =
    downed upstream — the signal _finish_episode promotes to MODEL_ERROR instead
    of a clean ENV_DONE."""
    from types import SimpleNamespace as NS

    from rllm.engine.agentflow_engine import _no_usable_model_output, _step_returned_nothing

    def step(content, tool_calls=None):
        return NS(model_output=NS(content=content), chat_completions=[{"role": "assistant", "content": content, "tool_calls": tool_calls}])

    assert _step_returned_nothing(step("")) is True
    assert _step_returned_nothing(step("ls -la")) is False
    assert _step_returned_nothing(step("", tool_calls=[{"id": "1"}])) is False  # tool-only turn is real work

    assert _no_usable_model_output(NS(trajectories=[NS(steps=[step(""), step("")])])) is True  # dead proxy
    assert _no_usable_model_output(NS(trajectories=[NS(steps=[])])) is True  # no LLM calls (the broken-eval case)
    assert _no_usable_model_output(NS(trajectories=[NS(steps=[step(""), step("echo hi")])])) is False  # partial — real work


def test_infra_taxonomy_membership_and_mapping():
    from rllm.types import INFRA_ERROR_REASONS, TerminationReason, termination_reason_from_error

    assert TerminationReason.MODEL_ERROR in INFRA_ERROR_REASONS
    assert termination_reason_from_error("DaytonaValidationError") == TerminationReason.SANDBOX_ERROR
    assert termination_reason_from_error("EmptyCompletion", default=TerminationReason.MODEL_ERROR) == TerminationReason.MODEL_ERROR
