"""Tests for the SdkWorkflow adapter and SdkWorkflowFactory."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rllm.agents.agent import Episode, Trajectory
from rllm.types import Step
from rllm.types import Trajectory as BaseTrajectory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_context(
    trace_id: str,
    session_name: str,
    session_uid: str,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
):
    """Build a minimal TraceContext-like object returned by SqliteTraceStore.

    The ``data`` dict must match what ``Trace(**data)`` expects:
    - ``input`` → ``LLMInput`` (messages, prompt_token_ids)
    - ``output`` → ``LLMOutput`` (message, finish_reason, output_token_ids, rollout_logprobs)
    """
    prompt_ids = prompt_ids or [1, 2, 3]
    completion_ids = completion_ids or [4, 5, 6]
    logprobs = logprobs or [-0.1, -0.2, -0.3]

    data = {
        "trace_id": trace_id,
        "session_name": session_name,
        "name": "test_model",
        "model": "test_model",
        "timestamp": time.time(),
        "latency_ms": 10.0,
        "tokens": {"prompt": len(prompt_ids), "completion": len(completion_ids), "total": len(prompt_ids) + len(completion_ids)},
        "metadata": {},
        "input": {
            "messages": [{"role": "user", "content": "hello"}],
            "prompt_token_ids": prompt_ids,
        },
        "output": {
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "output_token_ids": completion_ids,
            "rollout_logprobs": logprobs,
        },
    }

    tc = MagicMock()
    tc.id = trace_id
    tc.data = data
    return tc


# ---------------------------------------------------------------------------
# SdkWorkflow tests
# ---------------------------------------------------------------------------


class TestSdkWorkflow:
    """Unit tests for SdkWorkflow."""

    def _make_workflow(self, wrapped_func=None, store=None, **extra_kwargs):
        from rllm.experimental.engine.sdk_workflow import SdkWorkflow

        rollout_engine = MagicMock()
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)
        store = store or MagicMock()

        wf = SdkWorkflow(
            rollout_engine=rollout_engine,
            executor=executor,
            wrapped_func=wrapped_func or MagicMock(),
            store=store,
            **extra_kwargs,
        )
        return wf

    def test_is_multithread_safe(self):
        wf = self._make_workflow()
        assert wf.is_multithread_safe() is True

    @pytest.mark.asyncio
    async def test_run_float_return(self):
        """Agent returns a float reward -> Episode with grouped trajectories."""
        session_uid = "test-session-uid"

        async def mock_wrapped(metadata, **kwargs):
            return 1.0, session_uid

        store = AsyncMock()
        tc = _make_trace_context("tr1", "task1:0", session_uid)
        store.get_by_session_uid = AsyncMock(return_value=[tc])

        wf = self._make_workflow(wrapped_func=mock_wrapped, store=store)
        wf.reset(task={"question": "2+2?"}, uid="task1:0")

        episode = await wf.run(task={"question": "2+2?"}, uid="task1:0")

        assert isinstance(episode, Episode)
        assert episode.is_correct is True
        assert episode.id == "task1:0"
        assert len(episode.trajectories) >= 1
        assert episode.trajectories[0].reward == 1.0

    @pytest.mark.asyncio
    async def test_run_trajectory_view_return(self):
        """Agent returns List[BaseTrajectory] -> multi-trajectory Episode."""
        session_uid = "test-session-uid-2"

        traj_views = [
            BaseTrajectory(
                name="solver",
                steps=[Step(id="tr1", reward=0.5)],
                reward=0.5,
            ),
            BaseTrajectory(
                name="judge",
                steps=[Step(id="tr2", reward=1.0)],
                reward=1.0,
            ),
        ]

        async def mock_wrapped(metadata, **kwargs):
            return traj_views, session_uid

        tc1 = _make_trace_context("tr1", "task2:0", session_uid)
        tc2 = _make_trace_context("tr2", "task2:0", session_uid)
        store = AsyncMock()
        store.get_by_session_uid = AsyncMock(return_value=[tc1, tc2])

        wf = self._make_workflow(wrapped_func=mock_wrapped, store=store)
        wf.reset(task={"question": "solve x"}, uid="task2:0")

        episode = await wf.run(task={"question": "solve x"}, uid="task2:0")

        assert isinstance(episode, Episode)
        assert len(episode.trajectories) == 2
        assert episode.trajectories[0].name == "solver"
        assert episode.trajectories[1].name == "judge"
        assert episode.trajectories[1].reward == 1.0
        assert episode.is_correct is True

    @pytest.mark.asyncio
    async def test_run_tuple_return(self):
        """Agent returns (float, metrics_dict) -> Episode with metrics."""
        session_uid = "test-session-uid-3"

        async def mock_wrapped(metadata, **kwargs):
            return (0.0, {"custom_metric": 42}), session_uid

        store = AsyncMock()
        tc = _make_trace_context("tr1", "task3:0", session_uid)
        store.get_by_session_uid = AsyncMock(return_value=[tc])

        wf = self._make_workflow(wrapped_func=mock_wrapped, store=store)
        wf.reset(task={"question": "hard"}, uid="task3:0")

        episode = await wf.run(task={"question": "hard"}, uid="task3:0")

        assert isinstance(episode, Episode)
        assert episode.is_correct is False
        assert episode.metrics.get("custom_metric") == 42

    @pytest.mark.asyncio
    async def test_run_agent_failure_raises(self):
        """Agent function that raises should propagate as RuntimeError."""

        async def failing_func(metadata, **kwargs):
            raise ValueError("boom")

        store = AsyncMock()
        wf = self._make_workflow(wrapped_func=failing_func, store=store)
        wf.reset(task={}, uid="fail:0")

        with pytest.raises(RuntimeError, match="SDK agent function failed"):
            await wf.run(task={}, uid="fail:0")


# ---------------------------------------------------------------------------
# SdkWorkflowFactory tests
# ---------------------------------------------------------------------------


class TestSdkWorkflowFactory:
    """Unit tests for SdkWorkflowFactory."""

    def test_get_workflow_args(self):
        """Factory produces the expected keys for SdkWorkflow."""
        from rllm.experimental.engine.sdk_workflow import SdkWorkflowFactory

        with (
            patch.object(SdkWorkflowFactory, "_setup_proxy"),
            patch("rllm.experimental.engine.sdk_workflow.wrap_with_session_context", return_value=MagicMock()),
            patch("rllm.experimental.engine.sdk_workflow.SqliteTraceStore"),
        ):
            config = MagicMock()
            config.rllm = MagicMock()
            config.rllm.get.return_value = {}

            factory = SdkWorkflowFactory(
                agent_run_func=lambda: None,
                rollout_engine=MagicMock(),
                config=config,
            )

        args = factory.get_workflow_args()
        assert "wrapped_func" in args
        assert "store" in args
        assert "proxy_manager" in args
        assert "groupby_key" in args
        assert "traj_name_key" in args
        assert "sdk_cfg" in args


# ---------------------------------------------------------------------------
# UnifiedWorkflowEngine post_execute_hook tests
# ---------------------------------------------------------------------------


class TestPostExecuteHook:
    """Test that post_execute_hook is called after batch completion."""

    @pytest.mark.asyncio
    async def test_hook_invoked(self):
        """post_execute_hook should be called once after all tasks complete."""
        from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
        from rllm.workflows.workflow import Workflow

        hook_called = []

        async def hook():
            hook_called.append(True)

        # Create a minimal mock workflow class
        class DummyWorkflow(Workflow):
            async def run(self, task, uid, **kwargs):
                return Episode(id=uid, task=task, trajectories=[Trajectory(steps=[])])

            def is_multithread_safe(self):
                return True

        engine = UnifiedWorkflowEngine(
            workflow_cls=DummyWorkflow,
            workflow_args={},
            rollout_engine=MagicMock(),
            n_parallel_tasks=2,
            post_execute_hook=hook,
        )

        episodes = await engine.execute_tasks(
            tasks=[{"q": "a"}, {"q": "b"}],
            task_ids=["t1", "t2"],
        )

        assert len(episodes) == 2
        assert len(hook_called) == 1

    @pytest.mark.asyncio
    async def test_no_hook(self):
        """When no hook is provided, execution should still work fine."""
        from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
        from rllm.workflows.workflow import Workflow

        class DummyWorkflow(Workflow):
            async def run(self, task, uid, **kwargs):
                return Episode(id=uid, task=task, trajectories=[Trajectory(steps=[])])

            def is_multithread_safe(self):
                return True

        engine = UnifiedWorkflowEngine(
            workflow_cls=DummyWorkflow,
            workflow_args={},
            rollout_engine=MagicMock(),
            n_parallel_tasks=2,
        )

        episodes = await engine.execute_tasks(
            tasks=[{"q": "a"}],
            task_ids=["t1"],
        )

        assert len(episodes) == 1
