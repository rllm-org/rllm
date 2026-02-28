"""SdkWorkflow adapter: wraps an SDK agent_run_func behind the Workflow interface.

This module provides two classes:

- ``SdkWorkflowFactory`` – one-time setup of shared infrastructure (proxy, store,
  wrapped agent function).  Produces the kwargs dict that ``SdkWorkflow`` instances
  need.  Supports both Verl (``VerlProxyManager``) and Tinker
  (``TinkerProxyManager``) backends via the same LiteLLM proxy pipeline.

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
from rllm.sdk.protocol import Trace
from rllm.sdk.session.base import wrap_with_session_context
from rllm.sdk.store.sqlite_store import SqliteTraceStore
from rllm.types import Trajectory as BaseTrajectory
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
    - Sets up the LLM proxy (``VerlProxyManager`` for verl,
      ``TinkerProxyManager`` for tinker) so that ``get_chat_client()``
      inside the user function can reach the training model.
    - Creates a ``SqliteTraceStore`` for trace persistence.
    - Wraps ``agent_run_func`` with ``wrap_with_session_context()``.
    - Optionally creates a ``SandboxOrchestrator`` for sandboxed execution.
    - Exposes ``get_workflow_args()`` for ``SdkWorkflow`` construction.
    """

    def __init__(
        self,
        agent_run_func: Callable | None,
        rollout_engine: RolloutEngine,
        config: DictConfig,
    ) -> None:
        self.agent_run_func = agent_run_func
        self.rollout_engine = rollout_engine
        self.config = config
        self.rllm_config = config.rllm

        # SDK-specific config (may or may not exist – fall back to sensible defaults)
        self._sdk_cfg = self.rllm_config.get("sdk", {})

        # Sandbox config
        sandbox_cfg = self._sdk_cfg.get("sandbox", {})
        self._sandbox_enabled = sandbox_cfg.get("enabled", False)
        self.sandbox_orchestrator: Any = None
        self.agent_config: dict[str, Any] = {}

        # Trace store
        db_path = self._sdk_cfg.get("store", {}).get("path", None)
        self.store = SqliteTraceStore(db_path=db_path)

        # Grouping config
        processing_cfg = self._sdk_cfg.get("processing", {})
        self.groupby_key: str | None = processing_cfg.get("groupby_key", None)
        self.traj_name_key: str | None = processing_cfg.get("traj_name_key", None)

        # Wrapped agent function (adds session context) — None when sandbox-only
        if self.agent_run_func is not None:
            self.wrapped_func = wrap_with_session_context(
                self.agent_run_func,
                tracer_service_name="agent-sdk-worker",
            )
        else:
            self.wrapped_func = None

        # Set up sandbox orchestrator if enabled
        if self._sandbox_enabled:
            self._setup_sandbox(sandbox_cfg)

        # Proxy setup – depends on backend engine type
        self.proxy_manager: Any = None
        self._setup_proxy()

    # ----- sandbox setup -----

    def _setup_sandbox(self, sandbox_cfg: dict) -> None:
        """Create a SandboxOrchestrator from the config."""
        from rllm.sdk.sandbox import create_sandbox_orchestrator
        from rllm.sdk.sandbox.protocol import SandboxConfig

        sc = SandboxConfig(
            enabled=sandbox_cfg.get("enabled", False),
            backend=sandbox_cfg.get("backend", "local"),
            agent_dir=sandbox_cfg.get("agent_dir", ""),
            agent_module=sandbox_cfg.get("agent_module", "agent"),
            agent_func=sandbox_cfg.get("agent_func", "rollout"),
            image=sandbox_cfg.get("image", "python:3.11-slim"),
            dockerfile=sandbox_cfg.get("dockerfile", ""),
            requirements_file=sandbox_cfg.get("requirements_file", ""),
            install_rllm_sdk=sandbox_cfg.get("install_rllm_sdk", True),
            pool_mode=sandbox_cfg.get("pool_mode", "persistent"),
            num_workers=sandbox_cfg.get("num_workers", 8),
            worker_port=sandbox_cfg.get("worker_port", 8100),
            max_concurrent=sandbox_cfg.get("max_concurrent", 32),
            task_setup_commands=list(sandbox_cfg.get("task_setup_commands", [])),
            task_setup_timeout=sandbox_cfg.get("task_setup_timeout", 300),
            teardown_on_complete=sandbox_cfg.get("teardown_on_complete", True),
            execution_timeout=sandbox_cfg.get("execution_timeout", 600.0),
            setup_commands=list(sandbox_cfg.get("setup_commands", [])),
            extra=dict(sandbox_cfg.get("extra", {})),
        )

        self.sandbox_orchestrator = create_sandbox_orchestrator(sc)

        # Build agent config to pass into each execution
        model_name = self.rllm_config.get("model_name", "default")
        self.agent_config = {"model_id": model_name}
        self.agent_config.update(dict(sc.extra))

    async def initialize_sandbox(self) -> None:
        """Initialize the sandbox orchestrator (call after proxy is started)."""
        if self.sandbox_orchestrator is None:
            return

        from rllm.sdk.sandbox.result_store import ExecutionResultStore

        db_path = self._sdk_cfg.get("store", {}).get("path", None)
        result_store = ExecutionResultStore(db_path=db_path)
        proxy_url = self.proxy_manager.get_proxy_url() if self.proxy_manager else ""
        await self.sandbox_orchestrator.initialize(proxy_url, result_store)

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
                enable_result_store=self._sandbox_enabled,
            )
        elif proxy_mode == "external":
            self.proxy_manager.reload_proxy_config(config=config_payload)
        else:
            raise ValueError(f"Unknown proxy mode: {proxy_mode}")

        base_url = self.proxy_manager.get_proxy_url()
        os.environ["RLLM_SDK_BASE_URL"] = base_url
        logger.info("VerlProxyManager ready at %s", base_url)

    def _setup_tinker_proxy(self) -> None:
        """Set up TinkerProxyManager for tinker backend.

        Follows the same LiteLLM proxy pattern as ``_setup_verl_proxy()`` so
        that traces are captured by the ``TracingCallback`` instead of being
        stored directly by the inference server.
        """
        from rllm.sdk.proxy.proxy_manager import TinkerProxyManager

        proxy_cfg = self._sdk_cfg.get("proxy", {})
        model_name = self.rllm_config.get("model_name", "default")

        self.proxy_manager = TinkerProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_cfg.get("host", "127.0.0.1"),
            proxy_port=proxy_cfg.get("port", 4000),
            backend_host=proxy_cfg.get("backend_host", "127.0.0.1"),
            backend_port=proxy_cfg.get("backend_port", 8090),
            admin_token=proxy_cfg.get("admin_token", "my-shared-secret"),
            proxy_access_log=False,
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
                add_logprobs=True,
                add_return_token_ids=True,
                enable_result_store=self._sandbox_enabled,
            )
        elif proxy_mode == "external":
            self.proxy_manager.reload_proxy_config(config=config_payload)
        else:
            raise ValueError(f"Unknown proxy mode: {proxy_mode}")

        base_url = self.proxy_manager.get_proxy_url()
        os.environ["RLLM_SDK_BASE_URL"] = base_url
        logger.info("TinkerProxyManager ready at %s", base_url)

    # ----- public API -----

    def get_workflow_args(self) -> dict[str, Any]:
        """Return kwargs to pass into each ``SdkWorkflow`` via the engine."""
        args = {
            "wrapped_func": self.wrapped_func,
            "store": self.store,
            "proxy_manager": self.proxy_manager,
            "groupby_key": self.groupby_key,
            "traj_name_key": self.traj_name_key,
            "sdk_cfg": self._sdk_cfg,
        }
        if self.sandbox_orchestrator is not None:
            args["sandbox_orchestrator"] = self.sandbox_orchestrator
            args["agent_config"] = self.agent_config
        return args

    async def flush_traces_hook(self) -> None:
        """Batch-level trace flush – intended as ``post_execute_hook``."""
        if self.proxy_manager is not None and hasattr(self.proxy_manager, "flush_tracer"):
            await self.proxy_manager.flush_tracer(timeout=60.0)

    async def _close_store(self) -> None:
        """Close the trace store connection pool."""
        if self.store is not None:
            await self.store.close()

    def shutdown(self) -> None:
        """Release proxy / server resources."""
        # Shutdown sandbox orchestrator first
        if self.sandbox_orchestrator is not None:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.sandbox_orchestrator.shutdown())
                else:
                    loop.run_until_complete(self.sandbox_orchestrator.shutdown())
            except Exception:
                pass
            self.sandbox_orchestrator = None

        if self.proxy_manager is not None:
            self.proxy_manager.shutdown_proxy()
            self.proxy_manager = None
        # Best-effort close of the connection pool
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._close_store())
            else:
                loop.run_until_complete(self._close_store())
        except Exception:
            pass


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
        wrapped_func: Callable | None,
        store: SqliteTraceStore,
        proxy_manager: Any = None,
        groupby_key: str | None = None,
        traj_name_key: str | None = None,
        sdk_cfg: dict | None = None,
        sandbox_orchestrator: Any = None,
        agent_config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.wrapped_func = wrapped_func
        self.store = store
        self.proxy_manager = proxy_manager
        self.groupby_key = groupby_key
        self.traj_name_key = traj_name_key
        self.sdk_cfg = sdk_cfg or {}
        self.sandbox_orchestrator = sandbox_orchestrator
        self.agent_config = agent_config or {}

    def is_multithread_safe(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Workflow interface
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the SDK agent function and build an Episode from traces."""
        rollout_start_time = time.time()

        if self.sandbox_orchestrator is not None:
            return await self._run_sandboxed(task, uid, rollout_start_time)

        return await self._run_local(task, uid, rollout_start_time, **kwargs)

    async def _run_sandboxed(self, task: dict, uid: str, rollout_start_time: float) -> Episode:
        """Execute via sandbox orchestrator and build Episode from result."""
        from rllm.sdk.sandbox.protocol import ExecutionResult

        result: ExecutionResult = await self.sandbox_orchestrator.execute(task, self.agent_config)

        if not result.success:
            raise RuntimeError(f"Sandbox agent execution failed: {result.error}")

        # Collect traces from the store using the session_uid from the sandbox
        session_uid = result.session_uid
        traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)

        if not traces and self.proxy_manager is not None and hasattr(self.proxy_manager, "flush_tracer"):
            await self.proxy_manager.flush_tracer(timeout=10.0)
            traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)

        traces_for_session = [(tc.id, Trace(**tc.data)) for tc in traces]

        # Deserialize trajectories from the result
        trajectories_output = [BaseTrajectory(**t) for t in result.trajectories] if result.trajectories else []

        episode = self._build_episode(
            uid=uid,
            task=task,
            traces=traces_for_session,
            output=trajectories_output,
            metrics={},
            rollout_start_time=rollout_start_time,
        )
        return episode

    async def _run_local(self, task: dict, uid: str, rollout_start_time: float, **kwargs) -> Episode:
        """Execute the local agent function and build Episode from traces."""
        # Build metadata expected by wrap_with_session_context
        metadata: dict[str, Any] = {"session_name": uid, "task": task}

        # Execute the agent function
        success, output, session_uid = await self._execute_agent(metadata, task, **kwargs)

        if not success:
            # output is the traceback string on failure
            raise RuntimeError(f"SDK agent function failed: {output}")

        # Collect traces from the store (session_uid is unique per invocation,
        # so all returned traces already belong to this session)
        traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)

        # If no traces found, the proxy may have written them asynchronously.
        # Flush the proxy tracer and retry once (applies to clients like Strands
        # that don't do local tracing via get_chat_client()).
        if not traces and self.proxy_manager is not None and hasattr(self.proxy_manager, "flush_tracer"):
            await self.proxy_manager.flush_tracer(timeout=10.0)
            traces = await self.store.get_by_session_uid(session_uid, since=rollout_start_time)

        traces_for_session = [(tc.id, Trace(**tc.data)) for tc in traces]

        # Unpack output (may be float, list[Trajectory], or tuple)
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
            # List[Trajectory] – user-defined trajectory structure.
            # Steps are matched by ID first; if IDs don't match (e.g. Strands
            # hook provider generates different IDs than the proxy), fall back
            # to positional matching within each trajectory.
            assert all(isinstance(tv, BaseTrajectory) for tv in output)
            trajectories = []
            steps_by_position = list(steps)  # proxy-captured steps in order
            pos_offset = 0
            for tv in output:
                traj_steps = []
                for i, sv in enumerate(tv.steps):
                    step = step_id_to_step.get(sv.id)
                    if step is None:
                        # Positional fallback: proxy traces and user steps
                        # are in the same chronological order.
                        pos_idx = pos_offset + i
                        if pos_idx < len(steps_by_position):
                            step = steps_by_position[pos_idx]
                        else:
                            logger.warning("Step %s not found in store and no positional match – skipped", sv.id)
                            continue
                    step.reward = sv.reward
                    traj_steps.append(step)
                pos_offset += len(tv.steps)
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
