"""SdkWorkflow adapter: wraps an SDK agent_run_func behind the Workflow interface.

This module provides two classes:

- ``SdkWorkflowFactory`` – one-time setup of shared infrastructure (proxy, store,
  wrapped agent function).  Produces the kwargs dict that ``SdkWorkflow`` instances
  need.

- ``SdkWorkflow(Workflow)`` – per-task adapter executed by
  ``UnifiedWorkflowEngine``.  Each call to ``run()`` invokes the user-provided
  ``agent_run_func`` inside a session context, collects traces from the SQLite
  store, and converts them into an ``Episode``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from rllm.agents.agent import Episode, Trajectory
from rllm.sdk.data_process import group_steps, trace_to_step
from rllm.sdk.protocol import Trace, TrajectoryView
from rllm.sdk.session.base import wrap_with_session_context
from rllm.sdk.store.sqlite_store import SqliteTraceStore
from rllm.workflows.workflow import TerminationReason, Workflow

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.experimental.rollout import RolloutEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SdkWorkflowFactory – shared, one-time infrastructure setup
# ---------------------------------------------------------------------------


class SdkWorkflowFactory:
    """Initialise shared SDK infrastructure once and provide workflow args.

    Responsibilities:
    - Sets up the LLM proxy (VerlProxyManager for verl, InferenceAPIServer
      for tinker) so that ``get_chat_client()`` inside the user function can
      reach the training model.
    - Creates a ``SqliteTraceStore`` for trace persistence.
    - Wraps ``agent_run_func`` with ``wrap_with_session_context()``.
    - Exposes ``get_workflow_args()`` for ``SdkWorkflow`` construction.
    """

    def __init__(
        self,
        agent_run_func: Callable,
        rollout_engine: RolloutEngine,
        config: DictConfig,
    ) -> None:
        self.agent_run_func = agent_run_func
        self.rollout_engine = rollout_engine
        self.config = config
        self.rllm_config = config.rllm

        # SDK-specific config (may or may not exist – fall back to sensible defaults)
        self._sdk_cfg = self.rllm_config.get("sdk", {})

        # Trace store
        db_path = self._sdk_cfg.get("store", {}).get("path", None)
        self.store = SqliteTraceStore(db_path=db_path)

        # Grouping config
        processing_cfg = self._sdk_cfg.get("processing", {})
        self.groupby_key: str | None = processing_cfg.get("groupby_key", None)
        self.traj_name_key: str | None = processing_cfg.get("traj_name_key", None)

        # Wrapped agent function (adds session context)
        self.wrapped_func = wrap_with_session_context(
            self.agent_run_func,
            tracer_service_name="agent-sdk-worker",
        )

        # Proxy setup – depends on backend engine type
        self.proxy_manager: Any = None
        self._inference_server: Any = None
        self._setup_proxy()

    # ----- proxy setup helpers -----

    def _setup_proxy(self) -> None:
        """Detect engine type and start the appropriate proxy/server."""
        engine_cls_name = type(self.rollout_engine).__name__

        if engine_cls_name == "VerlEngine":
            self._setup_verl_proxy()
        elif engine_cls_name == "TinkerEngine":
            self._setup_tinker_proxy()
        else:
            logger.warning(
                "Unsupported rollout engine type '%s' for SDK proxy setup. SDK traces may not capture token IDs / logprobs.",
                engine_cls_name,
            )

    def _setup_verl_proxy(self) -> None:
        """Set up VerlProxyManager for verl backend."""
        from rllm.sdk.proxy.proxy_manager import VerlProxyManager

        proxy_cfg = self._sdk_cfg.get("proxy", {})
        model_name = self.config.actor_rollout_ref.model.path

        add_logprobs = getattr(self.config.algorithm, "rollout_correction", {}).get("rollout_is") is not None

        self.proxy_manager = VerlProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_cfg.get("host", "127.0.0.1"),
            proxy_port=proxy_cfg.get("port", 4000),
            admin_token=proxy_cfg.get("admin_token", "my-shared-secret"),
            proxy_access_log=False,
            add_logprobs=add_logprobs,
        )

        config_payload = self.proxy_manager.build_proxy_config()

        proxy_mode = proxy_cfg.get("mode", "subprocess")
        sync_tracer = proxy_cfg.get("sync_tracer", False)

        if proxy_mode == "subprocess":
            db_path = self._sdk_cfg.get("store", {}).get("path", None)
            project = self.rllm_config.trainer.get("project_name", "rllm-agent-sdk")
            self.proxy_manager.start_proxy_subprocess(
                config=config_payload,
                db_path=db_path,
                project=project,
                sync_tracer=sync_tracer,
                add_logprobs=add_logprobs,
                add_return_token_ids=True,
            )
        elif proxy_mode == "external":
            self.proxy_manager.reload_proxy_config(config=config_payload)
        else:
            raise ValueError(f"Unknown proxy mode: {proxy_mode}")

        base_url = self.proxy_manager.get_proxy_url()
        os.environ["RLLM_SDK_BASE_URL"] = base_url
        logger.info("VerlProxyManager ready at %s", base_url)

    def _setup_tinker_proxy(self) -> None:
        """Start InferenceAPIServer for tinker backend."""
        from rllm.experimental.remote.inference_server import (
            InferenceAPIServer,
            InferenceServerConfig,
        )

        proxy_cfg = self._sdk_cfg.get("proxy", {})
        host = proxy_cfg.get("host", "0.0.0.0")
        port = proxy_cfg.get("port", 8089)

        server_config = InferenceServerConfig(host=host, port=port)
        self._inference_server = InferenceAPIServer(
            rollout_engine=self.rollout_engine,
            config=server_config,
            trace_store=self.store,
        )
        self._inference_server.start()
        base_url = self._inference_server.inference_api_url
        os.environ["RLLM_SDK_BASE_URL"] = base_url
        logger.info("InferenceAPIServer for tinker started at %s (base_url=%s)", self._inference_server.url, base_url)

    # ----- public API -----

    def get_workflow_args(self) -> dict[str, Any]:
        """Return kwargs to pass into each ``SdkWorkflow`` via the engine."""
        return {
            "wrapped_func": self.wrapped_func,
            "store": self.store,
            "proxy_manager": self.proxy_manager,
            "groupby_key": self.groupby_key,
            "traj_name_key": self.traj_name_key,
            "sdk_cfg": self._sdk_cfg,
        }

    async def flush_traces_hook(self) -> None:
        """Batch-level trace flush – intended as ``post_execute_hook``."""
        if self.proxy_manager is not None and hasattr(self.proxy_manager, "flush_tracer"):
            await self.proxy_manager.flush_tracer(timeout=60.0)

    def shutdown(self) -> None:
        """Release proxy / server resources."""
        if self.proxy_manager is not None:
            self.proxy_manager.shutdown_proxy()
            self.proxy_manager = None
        if self._inference_server is not None:
            self._inference_server.stop()
            self._inference_server = None


# ---------------------------------------------------------------------------
# SdkWorkflow – per-task Workflow adapter
# ---------------------------------------------------------------------------


class SdkWorkflow(Workflow):
    """Workflow adapter that runs an SDK ``agent_run_func`` inside a session.

    Instantiated once per parallel slot by ``UnifiedWorkflowEngine``.
    Each ``run()`` call:

    1. Executes the wrapped agent function (with session context).
    2. Collects traces from SQLite after completion.
    3. Converts traces into an ``Episode``.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        wrapped_func: Callable,
        store: SqliteTraceStore,
        proxy_manager: Any = None,
        groupby_key: str | None = None,
        traj_name_key: str | None = None,
        sdk_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.wrapped_func = wrapped_func
        self.store = store
        self.proxy_manager = proxy_manager
        self.groupby_key = groupby_key
        self.traj_name_key = traj_name_key
        self.sdk_cfg = sdk_cfg or {}

    def is_multithread_safe(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the SDK agent function and build an Episode from traces."""
        rollout_start_time = time.time()

        # Build metadata expected by wrap_with_session_context
        metadata: dict[str, Any] = {"session_name": uid, "task": task}

        # Execute the agent function
        success, output, session_uid = await self._execute_agent(metadata, task, **kwargs)

        if not success:
            # output is the traceback string on failure
            raise RuntimeError(f"SDK agent function failed: {output}")

        # Collect traces from the store
        traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)
        traces_for_session: list[tuple[str, Trace]] = []
        for tc in traces:
            session_name = tc.data.get("session_name")
            if session_name != uid:
                continue
            traces_for_session.append((tc.id, Trace(**tc.data)))

        # Unpack output (may be float, list[TrajectoryView], or tuple)
        output_payload, metrics = self._unpack_output(output)

        # Build episode
        episode = self._build_episode(
            uid=uid,
            task=task,
            traces=traces_for_session,
            output=output_payload,
            metrics=metrics,
            rollout_start_time=rollout_start_time,
        )

        return episode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_agent(self, metadata: dict, task: dict, **kwargs) -> tuple[bool, Any, str | None]:
        """Run the wrapped agent function with exception handling."""
        try:
            if inspect.iscoroutinefunction(self.wrapped_func):
                output, session_uid = await self.wrapped_func(metadata, **task, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                bound = functools.partial(self.wrapped_func, metadata, **task, **kwargs)
                output, session_uid = await loop.run_in_executor(self.executor, bound)
            return True, output, session_uid
        except Exception:
            import traceback

            return False, traceback.format_exc(), None

    @staticmethod
    def _unpack_output(output: Any) -> tuple[Any, dict]:
        """Separate the payload from optional metrics dict."""
        if isinstance(output, tuple):
            assert len(output) == 2, "Tuple output must be (payload, metrics)"
            payload, metrics = output
            if isinstance(payload, float | int | bool):
                payload = float(payload)
            return payload, metrics
        if isinstance(output, float | int | bool):
            return float(output), {}
        if isinstance(output, list):
            return output, {}
        raise ValueError(f"Unsupported agent_run_func output type: {type(output)}")

    def _build_episode(
        self,
        uid: str,
        task: dict,
        traces: list[tuple[str, Trace]],
        output: Any,
        metrics: dict,
        rollout_start_time: float,
    ) -> Episode:
        """Convert traces + agent output into a fully-formed Episode."""
        steps = [trace_to_step(t) for _, t in traces]
        step_id_to_step = {tid: step for (tid, _), step in zip(traces, steps, strict=False)}

        if isinstance(output, float):
            trajectories = group_steps(steps, by=self.groupby_key, name_key=self.traj_name_key)
            for traj in trajectories:
                traj.reward = output
            is_correct = output >= 1.0
        elif isinstance(output, list):
            # List[TrajectoryView] – user-defined trajectory structure
            assert all(isinstance(tv, TrajectoryView) for tv in output)
            trajectories = []
            for tv in output:
                traj_steps = []
                for sv in tv.steps:
                    step = step_id_to_step.get(sv.id)
                    if step is None:
                        logger.warning("Step %s not found in store – skipped", sv.id)
                        continue
                    step.reward = sv.reward
                    traj_steps.append(step)
                trajectories.append(Trajectory(name=tv.name, steps=traj_steps, reward=tv.reward))
            is_correct = trajectories[-1].reward >= 1.0 if trajectories else False
        else:
            raise ValueError(f"Unexpected output type in _build_episode: {type(output)}")

        # Assemble metrics
        all_response_lens = [len(step.model_output.completion_ids) for traj in trajectories for step in traj.steps]
        all_prompt_lens = [len(step.model_output.prompt_ids) for traj in trajectories for step in traj.steps]
        metrics.update(
            {
                "empty": int(len(steps) == 0),
                "num_trajectories": len(trajectories),
                "steps_collected": len(traces),
                "steps_used": sum(len(t.steps) for t in trajectories),
                "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
                "max_response_len": max(all_response_lens, default=0),
                "min_response_len": min(all_response_lens, default=0),
                "max_prompt_len": max(all_prompt_lens, default=0),
                "min_prompt_len": min(all_prompt_lens, default=0),
            }
        )

        episode = Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=trajectories,
            metrics=metrics,
            termination_reason=TerminationReason.ENV_DONE,
        )
        return episode
