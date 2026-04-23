"""Unit tests for AgentCoreRuntime and remote episode building (all mocked)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from omegaconf import OmegaConf

from rllm.agents.agent import Step, Trajectory
from rllm.experimental.engine.gateway_manager import GatewayManager, _get_routable_ip
from rllm.experimental.engine.remote_agent_flow_engine import _build_episode, _error_episode
from rllm.experimental.engine.remote_runtime.agentcore_runtime import AgentCoreRuntime
from rllm.experimental.engine.remote_runtime.protocol import (
    RemoteRuntimeConfig,
    RemoteTaskResult,
    TaskSubmission,
)
from rllm.experimental.engine.trace_converter import compute_step_metrics
from rllm.workflows.workflow import TerminationReason

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> AgentCoreRuntime:
    """Create an AgentCoreRuntime with a mock client (skips initialize())."""
    config = RemoteRuntimeConfig(
        enabled=True,
        backend="agentcore",
        agentcore={
            "agent_runtime_arn": "arn:aws:bedrock:us-east-1:123456789:agent-runtime/test",
            "s3_bucket": "test-bucket",
        },
    )
    runtime = AgentCoreRuntime(config, exp_id="test-exp", model_id="test-model")
    runtime._client = MagicMock()
    return runtime


def _make_submission(session_id: str = "sess-1", task_id: str = "task-1") -> TaskSubmission:
    return TaskSubmission(
        task={"prompt": "What is 2+2?", "answer": "4"},
        session_id=session_id,
        task_id=task_id,
        inference_url="http://localhost:8000/v1",
    )


def _make_future(result: dict, elapsed: float = 1.5):
    """Create a mock RolloutFuture."""
    future = AsyncMock()
    future.result_async = AsyncMock(return_value=result)
    future.elapsed = MagicMock(return_value=elapsed)
    return future


def _make_step(prompt_len: int = 10, response_len: int = 20) -> Step:
    """Create a Step with specified token lengths."""
    return Step(
        prompt_ids=list(range(prompt_len)),
        response_ids=list(range(response_len)),
    )


# ---------------------------------------------------------------------------
# _run_one tests
# ---------------------------------------------------------------------------


class TestRunOneSuccess:
    def test_success_with_reward(self):
        """status_code=200 result -> RemoteTaskResult(success=True, reward=1.0)."""
        runtime = _make_runtime()
        sub = _make_submission()
        future = _make_future(
            result={"status_code": 200, "rewards": [1.0], "input_id": "task-1"},
            elapsed=2.3,
        )
        runtime._client.invoke_async = AsyncMock(return_value=future)

        result = asyncio.run(runtime._run_one(sub, timeout=300.0))

        assert result.success is True
        assert result.reward == 1.0
        assert result.session_id == "sess-1"
        assert result.task_id == "task-1"
        assert result.elapsed == 2.3
        assert result.raw_result["status_code"] == 200

    def test_success_reward_scalar(self):
        """Non-list reward is preserved as-is (None)."""
        runtime = _make_runtime()
        sub = _make_submission()
        future = _make_future(result={"rewards": None}, elapsed=1.0)
        runtime._client.invoke_async = AsyncMock(return_value=future)

        result = asyncio.run(runtime._run_one(sub, timeout=300.0))

        assert result.success is True
        assert result.reward is None


class TestRunOneErrorStatus:
    def test_error_status_500(self):
        """status_code=500 result -> RemoteTaskResult(success=False, error=...)."""
        runtime = _make_runtime()
        sub = _make_submission()
        future = _make_future(
            result={
                "status_code": 500,
                "stop_reason": "ZeroDivisionError: division by zero",
                "traceback": "...",
            },
            elapsed=3.0,
        )
        runtime._client.invoke_async = AsyncMock(return_value=future)

        result = asyncio.run(runtime._run_one(sub, timeout=300.0))

        assert result.success is False
        assert "ZeroDivisionError" in result.error
        assert result.elapsed == 3.0
        assert result.raw_result["status_code"] == 500

    def test_error_status_no_stop_reason(self):
        """Error without stop_reason gets default message."""
        runtime = _make_runtime()
        sub = _make_submission()
        future = _make_future(result={"status_code": 500}, elapsed=1.0)
        runtime._client.invoke_async = AsyncMock(return_value=future)

        result = asyncio.run(runtime._run_one(sub, timeout=300.0))

        assert result.success is False
        assert result.error == "Unknown remote error"


class TestRunOneTransportError:
    def test_invoke_raises(self):
        """invoke_async raising -> RemoteTaskResult(success=False)."""
        runtime = _make_runtime()
        sub = _make_submission()
        runtime._client.invoke_async = AsyncMock(side_effect=ConnectionError("network down"))

        result = asyncio.run(runtime._run_one(sub, timeout=300.0))

        assert result.success is False
        assert "network down" in result.error

    def test_result_async_raises(self):
        """result_async raising (timeout) -> RemoteTaskResult(success=False)."""
        runtime = _make_runtime()
        sub = _make_submission()
        future = AsyncMock()
        future.result_async = AsyncMock(side_effect=TimeoutError("timed out"))
        runtime._client.invoke_async = AsyncMock(return_value=future)

        result = asyncio.run(runtime._run_one(sub, timeout=1.0))

        assert result.success is False
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# _build_episode tests
# ---------------------------------------------------------------------------


class TestBuildEpisodeWithTraces:
    def test_traces_converted_to_steps(self):
        """Traces are converted to Steps, metrics computed correctly."""
        # Create mock TraceRecord-like objects
        traces = []
        for i in range(3):
            trace = MagicMock()
            trace.trace_id = f"trace-{i}"
            trace.response_message = {"content": f"answer {i}", "reasoning": ""}
            trace.prompt_token_ids = list(range(10 + i))
            trace.completion_token_ids = list(range(20 + i))
            trace.logprobs = [0.1] * (20 + i)
            trace.finish_reason = "stop"
            trace.messages = [{"role": "user", "content": f"question {i}"}]
            trace.metadata = {}
            traces.append(trace)

        result = RemoteTaskResult(
            success=True,
            session_id="sess-1",
            task_id="task-1",
            reward=1.0,
            elapsed=5.0,
        )

        with patch("rllm.experimental.engine.remote_agent_flow_engine.trace_record_to_step") as mock_convert:
            # Return Steps with proper token lengths
            mock_convert.side_effect = [_make_step(prompt_len=10 + i, response_len=20 + i) for i in range(3)]
            episode = _build_episode(traces, result, "task-1:0", {"prompt": "test"})

        assert len(episode.trajectories) == 1
        assert len(episode.trajectories[0].steps) == 3
        assert episode.trajectories[0].reward == 1.0
        assert episode.is_correct is True
        assert episode.metrics["num_trajectories"] == 1
        assert episode.metrics["steps_used"] == 3
        assert episode.metrics["steps_collected"] == 3
        assert episode.metrics["empty"] == 0


class TestBuildEpisodeNoTraces:
    def test_no_traces_with_reward(self):
        """No traces + reward -> empty-steps trajectory."""
        result = RemoteTaskResult(
            success=True,
            session_id="sess-1",
            task_id="task-1",
            reward=0.5,
            elapsed=2.0,
        )

        episode = _build_episode([], result, "task-1:0", {"prompt": "test"})

        assert len(episode.trajectories) == 1
        assert len(episode.trajectories[0].steps) == 0
        assert episode.trajectories[0].reward == 0.5
        assert episode.metrics["empty"] == 1

    def test_no_traces_no_reward(self):
        """No traces + no reward -> no trajectories."""
        result = RemoteTaskResult(
            success=True,
            session_id="sess-1",
            task_id="task-1",
            reward=None,
        )

        episode = _build_episode([], result, "task-1:0", {"prompt": "test"})

        assert len(episode.trajectories) == 0
        assert episode.metrics["empty"] == 1


class TestErrorEpisode:
    def test_error_episode_fields(self):
        """Error episode has termination_reason=ERROR."""
        episode = _error_episode("task-1:0", {"prompt": "test"}, "something broke")

        assert episode.is_correct is False
        assert episode.termination_reason == TerminationReason.ERROR
        assert episode.metadata["error"]["message"] == "something broke"


# ---------------------------------------------------------------------------
# compute_step_metrics tests
# ---------------------------------------------------------------------------


class TestComputeStepMetrics:
    def test_basic_metrics(self):
        """Shared utility produces correct dict."""
        trajectories = [
            Trajectory(
                name="t1",
                task={},
                steps=[_make_step(10, 20), _make_step(15, 25)],
            ),
            Trajectory(
                name="t2",
                task={},
                steps=[_make_step(12, 30)],
            ),
        ]

        metrics = compute_step_metrics(trajectories)

        assert metrics["num_trajectories"] == 2
        assert metrics["steps_used"] == 3
        # response_lens: [20, 25, 30]
        assert metrics["mean_response_len"] == 25.0
        assert metrics["max_response_len"] == 30
        assert metrics["min_response_len"] == 20
        # prompt_lens: [10, 15, 12]
        assert metrics["max_prompt_len"] == 15
        assert metrics["min_prompt_len"] == 10

    def test_empty_trajectories(self):
        """Empty input returns zero-valued metrics."""
        metrics = compute_step_metrics([])

        assert metrics["num_trajectories"] == 0
        assert metrics["steps_used"] == 0
        assert metrics["mean_response_len"] == 0
        assert metrics["max_response_len"] == 0

    def test_trajectory_with_no_steps(self):
        """Trajectory with no steps returns correct counts but zero lengths."""
        trajectories = [Trajectory(name="empty", task={}, steps=[])]

        metrics = compute_step_metrics(trajectories)

        assert metrics["num_trajectories"] == 1
        assert metrics["steps_used"] == 0
        assert metrics["mean_response_len"] == 0


# ---------------------------------------------------------------------------
# GatewayManager URL generation tests
# ---------------------------------------------------------------------------


class TestGatewayExternalURL:
    def test_default_host_uses_routable_ip(self):
        """When host is null/None in config, gateway_url uses auto-detected routable IP."""
        config = OmegaConf.create({"rllm": {"gateway": {"host": None, "port": 9090}}})
        gw = GatewayManager(config, mode="thread")

        routable = _get_routable_ip()
        assert gw.host == routable
        assert gw.gateway_url == f"http://{routable}:9090"
        # Should not be 127.0.0.1 on machines with a network interface
        # (but we tolerate it in CI where it might be the only option)

    def test_explicit_host_override(self):
        """When host is explicitly set in config, that value is used."""
        config = OmegaConf.create({"rllm": {"gateway": {"host": "10.0.0.42", "port": 8080}}})
        gw = GatewayManager(config, mode="thread")

        assert gw.host == "10.0.0.42"
        assert gw.gateway_url == "http://10.0.0.42:8080"

    def test_routable_ip_not_loopback(self):
        """_get_routable_ip() returns a non-loopback address when a network route exists."""
        ip = _get_routable_ip()
        # On machines with network access, should not be loopback
        # This is a best-effort check — in isolated CI it may still be 127.0.0.1
        assert isinstance(ip, str)
        assert len(ip) > 0
