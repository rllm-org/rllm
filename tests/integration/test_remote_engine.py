"""Integration tests for RemoteAgentFlowEngine + a passthrough gateway against live ACR.

Tests the full flow: gateway in passthrough mode (forwards to BASE_URL/vLLM)
-> RemoteAgentFlowEngine -> AgentCoreRuntime -> live ACR container -> gateway
traces.

Skipped unless all env vars are set: AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET,
AGENTCORE_BASE_URL, and AGENTCORE_MODEL_ID.

The gateway here runs in passthrough mode (no adapter), pointing at the vLLM
URL. We bypass ``GatewayManager`` since it's adapter-only — for an HTTP-only
upstream we wire ``create_app`` directly.
"""

from __future__ import annotations

import threading
import time
import uuid

import httpx
import pytest
import uvicorn

from rllm.experimental.engine.remote_agent_flow_engine import RemoteAgentFlowEngine
from rllm.experimental.engine.remote_runtime.agentcore_runtime import AgentCoreRuntime
from rllm.experimental.engine.remote_runtime.protocol import RemoteRuntimeConfig
from rllm.workflows.workflow import TerminationReason

from .conftest import AGENT_ARN, BASE_URL, MODEL_ID, S3_BUCKET, requires_agentcore

# ---------------------------------------------------------------------------
# Passthrough gateway shim
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _PassthroughGateway:
    """Minimal stand-in for GatewayManager that runs the gateway in passthrough
    mode. RemoteAgentFlowEngine only needs ``acreate_session``, ``get_session_url``,
    ``aget_traces``, and ``agent_api_key`` from the gateway interface.
    """

    def __init__(self, upstream_url: str):
        from rllm_model_gateway import (
            AsyncGatewayClient,
            GatewayClient,
            GatewayConfig,
            create_app,
            deserialize_extras,
        )
        from rllm_model_gateway.store.memory_store import MemoryTraceStore

        self._deserialize_extras = deserialize_extras

        port = _free_port()
        self.host = "127.0.0.1"
        self.port = port
        self._url = f"http://{self.host}:{self.port}"

        config = GatewayConfig(
            host=self.host,
            port=self.port,
            upstream_url=upstream_url.rstrip("/"),
            admin_api_key="integ-admin",
            agent_api_key="integ-agent",
        )
        self._config = config
        self._app = create_app(config, store=MemoryTraceStore())

        self._server = uvicorn.Server(uvicorn.Config(self._app, host=self.host, port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                if httpx.get(f"{self._url}/health", timeout=0.5).status_code == 200:
                    break
            except Exception:
                time.sleep(0.05)
        else:
            raise RuntimeError("passthrough gateway did not start")

        self._client = GatewayClient(self._url, api_key=config.admin_api_key)
        self._async_client = AsyncGatewayClient(self._url, api_key=config.admin_api_key)

    @property
    def agent_api_key(self) -> str:
        return self._config.agent_api_key

    def get_session_url(self, session_id: str) -> str:
        return f"{self._url}/sessions/{session_id}/v1"

    async def acreate_session(self, session_id: str, is_validation: bool = False) -> str:
        return await self._async_client.create_session(session_id=session_id)

    async def aget_traces(self, session_id: str, extras: bool = False):
        await self._async_client.flush()
        return await self._async_client.get_session_traces(session_id, extras=extras)

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._client.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway():
    """Module-scoped passthrough gateway pointing at the live vLLM upstream."""
    assert BASE_URL is not None, "AGENTCORE_BASE_URL must be set"
    gw = _PassthroughGateway(upstream_url=BASE_URL)
    yield gw
    gw.stop()


@pytest.fixture(scope="module")
def engine(gateway):
    """Module-scoped RemoteAgentFlowEngine backed by live ACR + passthrough gateway."""
    config = RemoteRuntimeConfig(
        enabled=True,
        backend="agentcore",
        agentcore={
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
        tasks = [{"prompt": GSM8K_PROBLEM, "answer": "694"}]
        episodes = await engine.execute_tasks(tasks, task_ids=["gsm-694"])

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep is not None
        assert ep.id == "gsm-694:0"
        assert len(ep.trajectories) >= 1
        traj = ep.trajectories[0]
        assert len(traj.steps) > 0
        # In passthrough mode we have no token IDs (vLLM doesn't return them
        # without explicit opt-in), so we only check the structured fields.
        for step in traj.steps:
            assert step.model_response is not None
        assert traj.reward is not None
        assert ep.metrics["steps_used"] > 0
        assert ep.metrics["steps_collected"] > 0


@requires_agentcore
class TestBatchTasksSessionCorrelation:
    @pytest.mark.asyncio
    async def test_batch_tasks_session_correlation(self, engine):
        tasks = [
            {"prompt": "What is 2 + 2?", "answer": "4"},
            {"prompt": "What is 10 * 5?", "answer": "50"},
            {"prompt": "What is 100 / 4?", "answer": "25"},
        ]
        task_ids = ["arith-4", "arith-50", "arith-25"]

        episodes = await engine.execute_tasks(tasks, task_ids=task_ids)

        assert len(episodes) == 3
        ep_ids = {ep.id for ep in episodes}
        assert ep_ids == {"arith-4:0", "arith-50:0", "arith-25:0"}
        for ep in episodes:
            assert ep is not None
            assert len(ep.trajectories) >= 1


@requires_agentcore
class TestTimeoutProducesErrorEpisode:
    @pytest.mark.asyncio
    async def test_timeout_produces_error_episode(self, engine):
        tasks = [{"prompt": "What is 1 + 1?", "answer": "2"}]

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
        tasks = [{"prompt": f"What is {i} + {i}?", "answer": str(2 * i)} for i in range(1, 9)]
        task_ids = [f"rate-{i}" for i in range(1, 9)]

        episodes = await engine.execute_tasks(tasks, task_ids=task_ids)

        assert len(episodes) == 8
        for ep in episodes:
            assert ep is not None
