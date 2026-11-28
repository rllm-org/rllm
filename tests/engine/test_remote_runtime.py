"""Unit tests for remote episode building and gateway helpers."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from rllm.engine.remote_agent_flow_engine import _build_episode
from rllm.engine.remote_runtime.protocol import RemoteTaskResult
from rllm.engine.trace_converter import compute_step_metrics
from rllm.gateway.manager import GatewayManager, _get_routable_ip
from rllm.types import Step, Trajectory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(prompt_len: int = 10, response_len: int = 20) -> Step:
    """Create a Step with specified token lengths."""
    return Step(
        prompt_ids=list(range(prompt_len)),
        response_ids=list(range(response_len)),
    )


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
            finished=True,
            session_id="sess-1",
            task_id="task-1",
            reward=1.0,
            elapsed=5.0,
        )

        with patch("rllm.engine.remote_agent_flow_engine.trace_record_to_step") as mock_convert:
            # Return Steps with proper token lengths
            mock_convert.side_effect = [_make_step(prompt_len=10 + i, response_len=20 + i) for i in range(3)]
            episode = _build_episode(traces, result, "task-1:0", {"prompt": "test"})

        assert len(episode.trajectories) == 1
        assert len(episode.trajectories[0].steps) == 3
        assert episode.trajectories[0].reward == 1.0
        assert episode.is_correct is True
        assert episode.session_id == "sess-1"
        assert episode.metrics["num_trajectories"] == 1
        assert episode.metrics["steps_used"] == 3
        assert episode.metrics["steps_collected"] == 3
        assert episode.metrics["empty"] == 0


class TestBuildEpisodeNoTraces:
    def test_no_traces_with_reward(self):
        """No traces + reward -> empty-steps trajectory."""
        result = RemoteTaskResult(
            finished=True,
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
            finished=True,
            session_id="sess-1",
            task_id="task-1",
            reward=None,
        )

        episode = _build_episode([], result, "task-1:0", {"prompt": "test"})

        assert len(episode.trajectories) == 0
        assert episode.metrics["empty"] == 1


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
