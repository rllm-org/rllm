"""Integration tests for RemoteAgentFlowEngine + GatewayManager against live ACR.

Tests the full flow: GatewayManager (auto-detected IP, dynamic port) ->
RemoteAgentFlowEngine -> AgentCoreRuntime -> live ACR container -> gateway traces.

Skipped unless all env vars are set: AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET,
AGENTCORE_BASE_URL, and AGENTCORE_MODEL_ID.
"""

import uuid

import pytest
from omegaconf import OmegaConf

from rllm.experimental.engine.gateway_manager import GatewayManager
from rllm.experimental.engine.remote_agent_flow_engine import RemoteAgentFlowEngine
from rllm.experimental.engine.remote_runtime.agentcore_runtime import AgentCoreRuntime
from rllm.experimental.engine.remote_runtime.protocol import RemoteRuntimeConfig
from rllm.workflows.workflow import TerminationReason

from .conftest import AGENT_ARN, BASE_URL, MODEL_ID, S3_BUCKET, requires_agentcore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway():
    """Module-scoped gateway with auto-detected host."""
    config = OmegaConf.create({"rllm": {"gateway": {"host": None, "port": 9090, "db_path": None}}})
    gw = GatewayManager(config, mode="thread")
    gw._start_thread()

    # Register the vLLM worker (strip /v1 suffix — gateway manages routing)
    assert BASE_URL is not None, "AGENTCORE_BASE_URL must be set"
    worker_url = BASE_URL.rstrip("/")
    if worker_url.endswith("/v1"):
        worker_url = worker_url[:-3]
    gw.client.add_worker(url=worker_url)

    yield gw
    gw.stop()


@pytest.fixture(scope="module")
def engine(gateway):
    """Module-scoped RemoteAgentFlowEngine backed by live ACR + gateway."""
    config = RemoteRuntimeConfig(
        enabled=True,
        backend="agentcore",
        backend_config={
            "agent_runtime_arn": AGENT_ARN,
            "s3_bucket": S3_BUCKET,
        },
    )
    assert MODEL_ID is not None, "AGENTCORE_MODEL_ID must be set"
    runtime = AgentCoreRuntime(config, exp_id=f"integ-engine-{uuid.uuid4().hex[:8]}", model_id=MODEL_ID)
    runtime.initialize()

    eng = RemoteAgentFlowEngine(runtime=runtime, gateway=gateway, session_timeout=300.0)
    yield eng
    runtime.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

GSM8K_PROBLEM = "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"


@requires_agentcore
class TestSingleTaskE2E:
    @pytest.mark.asyncio
    async def test_single_task_e2e(self, engine):
        """Full flow: engine creates session -> submits to ACR -> agent calls
        gateway for inference -> reward + traces -> Episode with steps."""
        tasks = [{"prompt": GSM8K_PROBLEM, "answer": "694"}]

        episodes = await engine.execute_tasks(tasks, task_ids=["gsm-694"])

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep is not None
        assert ep.id == "gsm-694:0"
        # Should have at least one trajectory with steps from gateway traces
        assert len(ep.trajectories) >= 1
        traj = ep.trajectories[0]
        assert len(traj.steps) > 0
        # Each step should have prompt and response token IDs
        for step in traj.steps:
            assert len(step.prompt_ids) > 0
            assert len(step.response_ids) > 0
        # Reward should be set
        assert traj.reward is not None
        # Metrics should be populated
        assert ep.metrics["steps_used"] > 0
        assert ep.metrics["steps_collected"] > 0


@requires_agentcore
class TestBatchTasksSessionCorrelation:
    @pytest.mark.asyncio
    async def test_batch_tasks_session_correlation(self, engine):
        """Submit 3 tasks -> verify 3 Episodes, each with correct session
        correlation and no cross-contamination of traces."""
        tasks = [
            {"prompt": "What is 2 + 2?", "answer": "4"},
            {"prompt": "What is 10 * 5?", "answer": "50"},
            {"prompt": "What is 100 / 4?", "answer": "25"},
        ]
        task_ids = ["arith-4", "arith-50", "arith-25"]

        episodes = await engine.execute_tasks(tasks, task_ids=task_ids)

        assert len(episodes) == 3
        # Each episode should have a unique ID matching its task
        ep_ids = {ep.id for ep in episodes}
        assert ep_ids == {"arith-4:0", "arith-50:0", "arith-25:0"}
        # Each episode should be non-None and have at least one trajectory
        for ep in episodes:
            assert ep is not None
            assert len(ep.trajectories) >= 1


@requires_agentcore
class TestTimeoutProducesErrorEpisode:
    @pytest.mark.asyncio
    async def test_timeout_produces_error_episode(self, engine):
        """Tiny timeout -> Episode with is_correct=False, termination_reason=ERROR."""
        tasks = [{"prompt": "What is 1 + 1?", "answer": "2"}]

        # Override timeout to be impossibly small
        original_timeout = engine.session_timeout
        engine.session_timeout = 0.01
        try:
            episodes = await engine.execute_tasks(tasks, task_ids=["timeout-test"])
        finally:
            engine.session_timeout = original_timeout

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.is_correct is False
        assert ep.termination_reason == TerminationReason.ERROR


@requires_agentcore
class TestBatchRateLimiting:
    @pytest.mark.asyncio
    async def test_batch_rate_limiting(self, engine):
        """Submit 8 tasks with tps_limit=5 -> all 8 results returned
        successfully (validates gather + rate limiter don't deadlock)."""
        tasks = [{"prompt": f"What is {i} + {i}?", "answer": str(2 * i)} for i in range(1, 9)]
        task_ids = [f"rate-{i}" for i in range(1, 9)]

        episodes = await engine.execute_tasks(tasks, task_ids=task_ids)

        assert len(episodes) == 8
        # All episodes should be non-None (no lost tasks)
        for ep in episodes:
            assert ep is not None
