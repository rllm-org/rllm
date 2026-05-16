import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rllm.data import Dataset
from rllm.experimental.buffer import TrajectoryGroupBuffer
from rllm.experimental.common.advantage import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.experimental.common.config import (
    AsyncTrainingConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.experimental.common.metrics import reduce_metrics_lists
from rllm.experimental.common.performance import simple_timer
from rllm.experimental.common.rejection_sampling import (
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
)
from rllm.experimental.common.transform import (
    _default_traj_grouping_hook,
    transform_episodes_to_trajectory_groups,
)
from rllm.experimental.common.visualization import print_metrics_table, visualize_trajectory_last_steps
from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
from rllm.experimental.metrics import MetricsAggregator
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.rollout import RolloutEngine
from rllm.experimental.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.types import Episode, TrajectoryGroup
from rllm.utils import EpisodeLogger, Tracking, extract_source_metadata
from rllm.workflows.store import Store
from rllm.workflows.workflow import TerminationReason, Workflow

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Common trainer state that's backend-agnostic. Reset at each training step."""

    rs_state: RejectionSamplingState = field(default_factory=RejectionSamplingState)
    global_step: int = 0
    epoch: int = 0
    total_steps: int = 0
    is_training: bool = True
    weight_version: int = 0
    # For timing and metrics
    timing_dict: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    extra_info: dict = field(default_factory=dict)
    # For passing the context
    episodes: list[Episode] | None = None
    trajectory_groups: list[TrajectoryGroup] | None = None
    backend_batch: Any | None = None

    def reset_batch(self) -> None:
        """Reset the trainer state for a new batch."""
        self.rs_state.reset()
        self.episodes = None
        self.trajectory_groups = None
        self.backend_batch = None

        self.timing_dict = {}
        self.metrics = {}
        self.extra_info = {}

    @property
    def has_episodes(self) -> bool:
        return self.episodes is not None and len(self.episodes) > 0

    @property
    def has_trajectory_groups(self) -> bool:
        return self.trajectory_groups is not None and len(self.trajectory_groups) > 0

    @property
    def has_backend_batch(self) -> bool:
        return self.backend_batch is not None


class UnifiedTrainer:
    """Unified trainer for backend-agnostic training.

    This trainer uses an async-prioritized design where the core pipeline methods
    are async. This accommodates backends that naturally use async operations
    (like Tinker) while still supporting sync backends.

    The main `fit()` method remains sync for ease of use, but internally runs
    the async training loop in a dedicated event loop thread.
    """

    def __init__(
        self,
        backend_cls: type[BackendProtocol],
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend_args: dict | None = None,
        *,
        traj_grouping_hook: Callable | None = None,
        traj_group_adv_estimator_map: dict | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """Initialize the UnifiedTrainer.

        Provide exactly one of ``workflow_class`` or (``agent_flow`` AND ``evaluator``).
        """
        has_agent_flow = kwargs.get("agent_flow") is not None and kwargs.get("evaluator") is not None
        remote_runtime_enabled = config.rllm.get("remote_runtime", {}).get("enabled", False)
        if not has_agent_flow and not remote_runtime_enabled:
            assert workflow_class is not None, "Either workflow_class, (agent_flow AND evaluator), or remote_runtime must be provided"

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.store = store
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # initializing and validating common configs
        self.config = config
        self.rllm_config = config.rllm

        # Read user-defined hooks from kwargs
        self.traj_grouping_hook = traj_grouping_hook or _default_traj_grouping_hook
        # Extract the TrajectoryGroup-specific estimator from kwargs
        self.traj_group_adv_estimator_map = traj_group_adv_estimator_map or {}

        # TODO(kylemontgomery1): disaggregate UnitifiedTrainer.__init__ from engine/infra setup

        self.backend = backend_cls(config=config, **(backend_args or {}))

        self._validate_and_setup_configs()
        self._setup_logging()

        self.async_config = AsyncTrainingConfig.from_config(self.rllm_config.get("async_training", {}))

        rollout_engine: RolloutEngine = self.backend.init_rollout_engine(
            cf_config=self.cf_config,
            transform_config=self.transform_config,
            rs_config=self.rs_config,
            algorithm_config=self.algorithm_config,
        )

        # Determine which engine path to use:
        # 1. agent_flow + evaluator → AgentFlowEngine (gateway-based, local)
        # 2. remote_runtime → RemoteAgentFlowEngine (gateway-based, remote)
        # 3. workflow_class → UnifiedWorkflowEngine (direct)
        self._gateway = None
        self._remote_runtime = None

        agent_flow = kwargs.get("agent_flow")
        evaluator = kwargs.get("evaluator")

        remote_runtime_cfg = self.rllm_config.get("remote_runtime", {})

        if agent_flow is not None and evaluator is not None:
            from rllm.engine.agentflow_engine import AgentFlowEngine
            from rllm.experimental.engine.gateway_manager import GatewayManager

            gateway_mode = "process" if kwargs.get("backend_name") == "verl" else "thread"
            self._gateway = GatewayManager(self.config, mode=gateway_mode)

            # merge the data.max_response_length into the sampling_params; TODO(listar2000): refactor the data config group
            training_sampling_params = {
                **OmegaConf.to_container(self.rllm_config.rollout.train, resolve=True),
                "max_tokens": self.config.data.max_response_length,
            }
            val_sampling_params = {
                **OmegaConf.to_container(self.rllm_config.rollout.val, resolve=True),
                "max_tokens": self.config.data.max_response_length,
            }

            self.agent_workflow_engine = AgentFlowEngine(
                agent_flow=agent_flow,
                evaluator=evaluator,
                gateway=self._gateway,
                model=self.config.get("model", {}).get("name", "default"),
                n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
                retry_limit=self.rllm_config.workflow.retry_limit,
                raise_on_error=self.rllm_config.workflow.get("raise_on_error", True),
                episode_logger=self.episode_logger,
                train_sampling_params=training_sampling_params,
                val_sampling_params=val_sampling_params,
            )
        elif remote_runtime_cfg.get("enabled", False):
            from rllm.experimental.engine.gateway_manager import GatewayManager
            from rllm.experimental.engine.remote_agent_flow_engine import (
                RemoteAgentFlowEngine,
            )
            from rllm.experimental.engine.remote_runtime import (
                RemoteRuntimeConfig,
                create_remote_runtime,
            )

            gateway_mode = "process" if kwargs.get("backend_name") == "verl" else "thread"
            self._gateway = GatewayManager(self.config, mode=gateway_mode)

            remote_runtime_config = RemoteRuntimeConfig(
                enabled=True,
                backend=remote_runtime_cfg.get("backend", "agentcore"),
                agentcore=dict(remote_runtime_cfg.get("agentcore", {})),
                harbor=dict(remote_runtime_cfg.get("harbor", {})),
                session_timeout=remote_runtime_cfg.get("session_timeout", 900.0),
            )
            self._remote_runtime = create_remote_runtime(
                remote_runtime_config,
                exp_id=self.rllm_config.trainer.experiment_name,
                model_id=self.config.get("model", {}).get("name", "default"),
            )

            self.agent_workflow_engine = RemoteAgentFlowEngine(
                runtime=self._remote_runtime,
                gateway=self._gateway,
                session_timeout=remote_runtime_config.session_timeout,
                n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
                episode_logger=self.episode_logger,
            )
        else:
            self.agent_workflow_engine = UnifiedWorkflowEngine(
                workflow_cls=self.workflow_class,
                workflow_args=self.workflow_args,
                rollout_engine=rollout_engine,
                config=self.config,
                n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
                retry_limit=self.rllm_config.workflow.retry_limit,
                raise_on_error=self.rllm_config.workflow.raise_on_error,
                episode_logger=self.episode_logger,
                store=self.store,
            )

        self.tokenizer = None
        if hasattr(self.backend, "tokenizer"):
            self.tokenizer = self.backend.tokenizer

    def _validate_and_setup_configs(self):
        """Validate and setup common configs."""
        # validate common, backend-agnostic configs
        assert self.rllm_config is not None, "rLLM config is not set"

        if self.rllm_config.rejection_sample.multiplier != 1:
            assert self.rllm_config.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # validate backend-specific configs
        self.backend.validate_config()

        self.cf_config = CompactFilteringConfig.from_config(self.rllm_config.compact_filtering)
        self.transform_config = TransformConfig.from_config(
            self.rllm_config.get("transform", {}),
            broadcast=self.rllm_config.stepwise_advantage.mode == "broadcast",
        )
        self.rs_config = RejectionSamplingConfig.from_config(self.rllm_config.rejection_sample)
        self.algorithm_config = AlgorithmConfig.from_config(
            self.rllm_config.algorithm,
            stepwise_advantage_mode=self.rllm_config.stepwise_advantage.mode,
            estimator_map=self.traj_group_adv_estimator_map,
        )

    def _setup_logging(self):
        """Setup up both the tracking and episode logging."""
        # create episode logger if enabled in config
        self.episode_logger = None
        if self.rllm_config.episode_logging.get("log_episodes", False):
            episode_log_dir = self.rllm_config.episode_logging.get(
                "episode_log_dir",
                f"logs/{self.rllm_config.trainer.project_name}/{self.rllm_config.trainer.experiment_name}",
            )
            self.episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        source_metadata = extract_source_metadata(
            workflow_class=self.workflow_class,
            workflow_args=self.workflow_args,
        )

        self.logger = Tracking(
            project_name=self.rllm_config.trainer.project_name,
            experiment_name=self.rllm_config.trainer.experiment_name,
            default_backend=self.rllm_config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            source_metadata=source_metadata,
        )

    # =========================================================================
    # Main training loop methods
    # =========================================================================

    # TODO(kylemontgomery1): better seperation of on policy vs fully async training code

    def fit(self):
        """Main training loop (sync entry point)."""
        asyncio.run(self.fit_async())

    async def fit_async(self) -> None:
        """Public async entry point for the full training process."""
        # Initialize remote runtime (if enabled) before the workflow pool
        if self._remote_runtime is not None:
            self._remote_runtime.initialize()

        # initialize the UnifiedWorkflowEngine (init the workflow pool)
        # AgentFlowEngine and RemoteAgentFlowEngine don't need pool initialization
        if hasattr(self.agent_workflow_engine, "initialize_pool"):
            await self.agent_workflow_engine.initialize_pool()

        trainer_state = TrainerState()

        await self.backend.on_train_start(trainer_state)

        if hasattr(self, "_gateway") and self._gateway is not None:
            self._gateway.start(self.backend.rollout_engine)

        if self.rllm_config.trainer.get("val_before_train", True):
            await self._validate_async(trainer_state)
            if self.rllm_config.trainer.get("val_only", False):
                return

        # we start from step (1 + original start batch index)
        trainer_state.global_step += 1

        try:
            await self._fit_async(trainer_state)
        finally:
            try:
                await self.backend.on_train_end(trainer_state)
            except Exception:
                logger.exception(f"{self.backend.__class__.__name__}.on_train_end() failed during fit_async() cleanup")

    async def _fit_async(self, trainer_state: TrainerState) -> None:
        """Dispatch to sync or concurrent training based on config."""
        # TODO(listar2000): after some benchmarking, maybe we just keep the fully-async and treat on-policy as a special case.
        if self.async_config.enable:
            await self._fit_fully_async(trainer_state)
        else:
            await self._fit_on_policy(trainer_state)

    async def _fit_on_policy(self, trainer_state: TrainerState) -> None:
        """Synchronous training loop (the most vanilla, standalone case that does not support minibatching or off-policy training)."""
        # TODO(kylemontgomery1): dataloader should be backend-agnostic
        train_dataloader: Iterable = self.backend.get_dataloader(self.train_dataset, trainer_state)
        break_via_total_batches = False  # used to break the training loop via the `total_batches` parameter
        use_total_batches = self.rllm_config.trainer.get("total_batches") is not None and self.rllm_config.trainer.total_batches > 0

        if use_total_batches:
            trainer_state.total_steps = self.rllm_config.trainer.total_batches
        else:
            trainer_state.total_steps = len(train_dataloader) * self.rllm_config.trainer.total_epochs

        for epoch in range(self.rllm_config.trainer.total_epochs):
            # recursively break through the outer loop
            if break_via_total_batches:
                break

            pprint(f"epoch {epoch}, step {trainer_state.global_step} started")
            trainer_state.epoch = epoch
            await self.backend.on_epoch_start(trainer_state)

            for batch in train_dataloader:
                trainer_state.reset_batch()

                await self.backend.on_batch_start(trainer_state)
                with simple_timer("step", trainer_state.timing_dict):
                    await self._train_batch_async(batch, trainer_state)
                await self.backend.on_batch_end(trainer_state)

                print_metrics_table(trainer_state.metrics, trainer_state.global_step)
                self.logger.log(
                    data=trainer_state.metrics,
                    step=trainer_state.global_step,
                    episodes=trainer_state.episodes,
                    trajectory_groups=trainer_state.trajectory_groups,
                )

                # if the config specifies the `total_batches` parameter, then we check if we should stop
                if use_total_batches and trainer_state.global_step >= self.rllm_config.trainer.total_batches:
                    break_via_total_batches = True
                    break

                # periodic validation
                if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                    await self._validate_async(trainer_state)

                trainer_state.global_step += 1

            await self.backend.on_epoch_end(trainer_state)

        # final validation after training
        if self.rllm_config.trainer.test_freq > 0:
            await self._validate_async(trainer_state)

    async def _train_batch_async(self, batch: Any, trainer_state: TrainerState) -> None:
        """Train a batch (async implementation)."""
        self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="train", epoch=trainer_state.epoch)

        # TODO(kylemontgomery1): episode generation should be backend-agnostic
        # stage 1: generate episodes (async) and collect metrics (sync)
        trainer_state.episodes = await self.backend.generate_episodes(batch, agent_workflow_engine=self.agent_workflow_engine, is_validation=False)
        if not trainer_state.has_episodes:
            return

        workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(trainer_state.episodes)
        for key, value in workflow_metrics.items():
            trainer_state.metrics[f"batch/{key}"] = np.mean(value)

        total_counts = max(sum(termination_counts.values()), 1)
        for r in TerminationReason:
            trainer_state.metrics[f"batch/termination_reason/{r.value}"] = termination_counts[r.value] / total_counts

        # stage 2: transform episodes to trajectory groups (sync)
        trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(trainer_state.episodes, self.transform_config, self.cf_config, traj_grouping_hook=self.traj_grouping_hook)
        trainer_state.trajectory_groups = trajectory_groups
        trainer_state.metrics.update(transform_metrics)

        # stage 3: apply rejection sampling (sync)
        filtered_groups, filtered_episodes, rs_metrics = apply_rejection_sampling_and_filtering(
            trainer_state.episodes,
            trainer_state.trajectory_groups,
            self.rs_config,
            trainer_state.rs_state,
        )
        trainer_state.metrics.update(rs_metrics)
        trainer_state.trajectory_groups = filtered_groups
        trainer_state.episodes = filtered_episodes
        if not trainer_state.has_trajectory_groups:
            return

        # stage 4: transform rllm-native data structures to backend-specific format (sync)
        backend_batch = self.backend.transform_to_backend_batch(trainer_state)
        trainer_state.backend_batch = backend_batch

        # stage 5: process backend batch (async) - compute log probs, critic values, etc.
        await self.backend.process_backend_batch(trainer_state)
        assert trainer_state.has_backend_batch, "Backend batch is not transformed or processed successfully"

        # TODO(kylemontgomery1): compute advantages should be backend-agnostic
        # stage 6: compute advantages (async)
        await self.backend.compute_advantages(trainer_state, self.algorithm_config)

        # stage 7: update policy (async)
        await self.backend.update_policy(trainer_state)

        # stage 8: cleanup, logging, visualization, etc. (sync)
        if self.tokenizer is not None:
            visualize_trajectory_last_steps(
                trainer_state.trajectory_groups,
                tokenizer=self.tokenizer,
                max_steps_to_visualize=2,
                show_workflow_metadata=True,
            )

    # =========================================================================
    # Fully-asynchronous training pipeline
    # =========================================================================

    async def _fit_fully_async(self, trainer_state: TrainerState) -> None:
        """Fully-async generation + training with group-level streaming."""
        assert self.config.data.train_batch_size == 1, f"Async training requires train_batch_size=1, got {self.config.data.train_batch_size}"
        assert not getattr(self.agent_workflow_engine, "raise_on_error", False), "Async training requires raise_on_error=False so that process_task_with_retry always returns an episode"
        coord_config = SyncCoordinatorConfig(
            mini_batch_size=self.async_config.mini_batch_size,
            group_size=self.rllm_config.rollout.n,
            staleness_threshold=self.async_config.staleness_threshold,
            trigger_parameter_sync_step=self.async_config.trigger_parameter_sync_step,
        )
        coordinator = SyncCoordinator(coord_config)
        aggregator = MetricsAggregator()
        buffer = TrajectoryGroupBuffer(
            group_size=self.rllm_config.rollout.n,
            coordinator=coordinator,
            aggregator=aggregator,
            algorithm_config=self.algorithm_config,
            transform_config=self.transform_config,
            cf_config=self.cf_config,
            rs_config=self.rs_config,
            episode_offload_dir=self.async_config.episode_offload_dir,
            trajectory_group_offload_dir=self.async_config.trajectory_group_offload_dir,
        )

        # Compute total_steps for LR scheduling
        train_dataloader = self.backend.get_dataloader(self.train_dataset, trainer_state)
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        if use_total_batches:
            trainer_state.total_steps = self.rllm_config.trainer.total_batches
        else:
            trainer_state.total_steps = len(train_dataloader) * self.rllm_config.trainer.total_epochs

        total_tasks = len(train_dataloader) * self.rllm_config.trainer.total_epochs
        pbar = tqdm(total=total_tasks, desc="Tasks", unit="task")
        buffer._pbar = pbar

        try:
            gen_task = asyncio.create_task(self._generation_loop(trainer_state, buffer, coordinator))
            await self._training_loop(trainer_state, buffer, coordinator, aggregator)
            if not gen_task.done():
                gen_task.cancel()
                try:
                    await gen_task
                except asyncio.CancelledError:
                    pass
        finally:
            pbar.close()

    async def _generation_loop(
        self,
        trainer_state: TrainerState,
        buffer: TrajectoryGroupBuffer,
        coordinator: SyncCoordinator,
    ) -> None:
        """Generate episodes and stream to TrajectoryGroupBuffer."""
        group_size = self.rllm_config.rollout.n

        try:
            for epoch in range(self.rllm_config.trainer.total_epochs):
                await self.backend.on_epoch_start(trainer_state)
                train_dataloader = self.backend.get_dataloader(self.train_dataset, trainer_state)
                self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="train", epoch=epoch)

                for batch in train_dataloader:
                    task = batch[0]

                    await coordinator.wait_for_generation_allowed()
                    if not coordinator.has_quota():
                        await coordinator.wait_for_throttle()
                    coordinator.on_group_dispatched()

                    task_id = str(uuid.uuid4())
                    for rollout_idx in range(group_size):

                        async def _run_rollout(t=task, tid=task_id, ridx=rollout_idx):
                            _, _, _, episode = await self.agent_workflow_engine.process_task_with_retry(task=t, task_id=tid, rollout_idx=ridx, result_idx=0)
                            await buffer.add_episode(tid, episode)

                        t = asyncio.create_task(_run_rollout())
                        coordinator.track_task(t)

                await self.backend.on_epoch_end(trainer_state)

            await coordinator.wait_for_drain()
        finally:
            buffer.mark_generation_complete()

    async def _training_loop(
        self,
        trainer_state: TrainerState,
        buffer: TrajectoryGroupBuffer,
        coordinator: SyncCoordinator,
        aggregator: MetricsAggregator,
    ) -> None:
        """Consume task batches from buffer, run forward-backward + optimizer step."""
        mini_batch_size = self.async_config.mini_batch_size
        fwd_bwd_group_size = self.async_config.fwd_bwd_group_size
        num_fwd_bwd_passes = mini_batch_size // fwd_bwd_group_size
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        rollout_engine = getattr(self.agent_workflow_engine, "rollout_engine", None)

        while True:
            trainer_state.reset_batch()
            step_start = time.perf_counter()
            weight_versions = []
            all_trajectory_groups: list[TrajectoryGroup] = []
            all_episodes: list[Episode] = []
            groups_consumed = 0
            buffer_wait_time = 0.0
            done = False

            buffered = buffer._queue.qsize()
            logger.info(
                f"[TrainingLoop] Step {trainer_state.global_step}: waiting for {mini_batch_size} task batches ({num_fwd_bwd_passes} fwd-bwd passes x {fwd_bwd_group_size} groups), {buffered} buffered"
            )

            # 1. Pull mini_batch_size task batches total, split into
            #    num_fwd_bwd_passes forward-backward passes of fwd_bwd_group_size each.
            for pass_idx in range(num_fwd_bwd_passes):
                chunk_groups: list[TrajectoryGroup] = []

                for _ in range(fwd_bwd_group_size):
                    t_wait = time.perf_counter()
                    task_batch = await buffer.get()
                    buffer_wait_time += time.perf_counter() - t_wait
                    if task_batch is None:
                        done = True
                        break

                    coordinator.on_group_consumed()
                    groups_consumed += 1

                    for group in task_batch.groups:
                        weight_versions.append(group.weight_version)
                    chunk_groups.extend(task_batch.groups)
                    all_trajectory_groups.extend(task_batch.groups)
                    all_episodes.extend(task_batch.episodes)

                if not chunk_groups or done:
                    break

                # Forward-backward on this chunk
                trainer_state.trajectory_groups = chunk_groups

                if trainer_state.has_trajectory_groups:
                    logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: fwd-bwd pass {pass_idx + 1}/{num_fwd_bwd_passes} ({len(chunk_groups)} groups)")
                    await self.backend.on_batch_start(trainer_state)
                    trainer_state.backend_batch = self.backend.transform_to_backend_batch(trainer_state)
                    await self.backend.process_backend_batch(trainer_state)

                    # Drain per-chunk backend metrics into aggregator
                    aggregator.record_dict(trainer_state.metrics)
                    trainer_state.metrics = {}

            # Only run optimizer step on a full batch
            if groups_consumed < mini_batch_size:
                logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: incomplete batch ({groups_consumed}/{mini_batch_size}), stopping")
                break

            # 2. Optimizer step
            logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: optimizer step")
            await self.backend.update_policy(trainer_state)

            # 3. Capture pre-sync metrics (before weight sync resets coordinator state)
            staleness_values = [coordinator.weight_version - v for v in weight_versions]
            aggregator.record("async/staleness_mean", float(np.mean(staleness_values)))
            aggregator.record("async/staleness_min", float(np.min(staleness_values)))
            aggregator.record("async/staleness_max", float(np.max(staleness_values)))
            aggregator.record("async/groups_consumed", groups_consumed)
            aggregator.record("time/buffer_wait", buffer_wait_time)
            pre_sync_coordinator_stats = coordinator.stats()
            pre_sync_buffer_stats = buffer.stats()

            # 4. Weight sync
            coordinator.on_training_step_complete()
            sync_time = 0.0
            if coordinator.should_sync():
                logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: triggering weight sync")
                t0 = time.perf_counter()
                await self._perform_weight_sync(trainer_state, coordinator, rollout_engine)
                sync_time = time.perf_counter() - t0
                logger.info(f"[TrainingLoop] Step {trainer_state.global_step}: weight sync complete ({sync_time:.2f}s)")
            if sync_time > 0:
                aggregator.record("time/weight_sync", sync_time)
            aggregator.record("time/step", time.perf_counter() - step_start)

            # Set all trajectory groups and stripped episodes for visualization/logging
            trainer_state.trajectory_groups = all_trajectory_groups
            trainer_state.episodes = all_episodes

            if self.tokenizer is not None and trainer_state.has_trajectory_groups:
                visualize_trajectory_last_steps(
                    trainer_state.trajectory_groups,
                    tokenizer=self.tokenizer,
                    max_steps_to_visualize=2,
                    show_workflow_metadata=True,
                )

            # 5. Flush aggregator and merge pre-sync snapshots into trainer_state.metrics
            trainer_state.metrics.update(aggregator.flush())
            trainer_state.metrics.update(pre_sync_buffer_stats)
            trainer_state.metrics.update(pre_sync_coordinator_stats)

            # 6. Compute derived metrics
            step_time = trainer_state.metrics.get("time/step", 1.0)
            trainer_state.metrics["async/trainer_idle_ratio"] = buffer_wait_time / max(step_time, 1e-9)

            # 7. on_batch_end writes backend metrics (progress, optim, timing)
            await self.backend.on_batch_end(trainer_state)

            # 7. Print and log
            print_metrics_table(trainer_state.metrics, trainer_state.global_step)
            self.logger.log(
                data=trainer_state.metrics,
                step=trainer_state.global_step,
                episodes=trainer_state.episodes,
                trajectory_groups=trainer_state.trajectory_groups,
            )

            # Periodic validation
            if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                await self._validate_async_with_pause(trainer_state, coordinator)

            trainer_state.global_step += 1

            if use_total_batches and trainer_state.global_step >= self.rllm_config.trainer.total_batches:
                break

    async def _perform_weight_sync(self, trainer_state: TrainerState, coordinator: SyncCoordinator, rollout_engine: RolloutEngine | None) -> None:
        """Synchronize weights between training and rollout engines."""
        if not self.async_config.partial_rollout:
            coordinator.pause_generation()
            await coordinator.wait_for_drain()

        trainer_state.weight_version = coordinator.weight_version + 1
        await self.backend.on_policy_updated(trainer_state)
        if rollout_engine is not None:
            rollout_engine.weight_version = trainer_state.weight_version
        coordinator.on_sync_complete()

        if not self.async_config.partial_rollout:
            coordinator.resume_generation()

    async def _validate_async_with_pause(self, trainer_state: TrainerState, coordinator: SyncCoordinator) -> dict:
        """Validation with dispatch-level pause. Waits for workflows to drain, then runs validation."""
        coordinator.pause_generation()
        await coordinator.wait_for_drain()
        try:
            return await self._validate_async(trainer_state)
        finally:
            coordinator.resume_generation()

    async def _validate_async(self, trainer_state: TrainerState) -> dict:
        """Validate the model (async implementation)."""
        n_val_samples = self.rllm_config.rollout.n_val
        val_metrics = defaultdict(list)

        if not await self.backend.on_validation_start(trainer_state):
            return {}
        # manually manage the testing time
        test_begin = time.perf_counter()
        self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="val", epoch=trainer_state.epoch)

        is_correct_lst, uid_lst, data_source_lst = [], [], []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

        val_dataloader: Iterable = self.backend.get_dataloader(self.val_dataset, trainer_state)
        for batch in val_dataloader:
            # Generate episodes and transform to trajectory groups
            val_episodes = await self.backend.generate_episodes(batch, agent_workflow_engine=self.agent_workflow_engine, is_validation=True)
            val_trajectory_groups, _ = transform_episodes_to_trajectory_groups(val_episodes, self.transform_config, self.cf_config, traj_grouping_hook=self.traj_grouping_hook)
            reward_metrics = collect_reward_and_advantage_from_trajectory_groups(val_trajectory_groups, self.algorithm_config, collect_advantage=False)

            is_correct_lst.extend([episode.is_correct for episode in val_episodes])
            uid_lst.extend([episode.task_id for episode in val_episodes])

            data_sources = [episode.info.get("data_source", "unknown") for episode in val_episodes]
            data_source_lst.extend(data_sources)

            for episode, data_source in zip(val_episodes, data_sources, strict=True):
                for key, value in episode.metrics.items():
                    # episode.metrics can contain non-numeric values -- skip in the workflow metrics.
                    try:
                        workflow_metrics_by_source[data_source][key].append(float(value))
                    except (TypeError, ValueError):
                        continue

            for key, value in reward_metrics.items():
                val_metrics[f"val/{key}"].append(value)

        test_end = time.perf_counter()
        val_metrics["time/testing"] = test_end - test_begin
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            val_metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            val_metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if values:
                        val_metrics[f"val/{data_source}/{key}"] = np.mean(values)

        # post-process the val metrics to reduce any "list values" into scalars
        reduce_metrics_lists(val_metrics)
        print_metrics_table(val_metrics, trainer_state.global_step, title="Validation")
        self.logger.log(data=val_metrics, step=trainer_state.global_step)
        await self.backend.on_validation_end(trainer_state)
        return val_metrics

    def shutdown(self):
        """Shutdown the trainer and cleanup resources."""
        if hasattr(self, "_remote_runtime") and self._remote_runtime is not None:
            self._remote_runtime.shutdown()
            self._remote_runtime = None
        if hasattr(self, "_gateway") and self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
        if hasattr(self, "agent_workflow_engine") and self.agent_workflow_engine is not None:
            self.agent_workflow_engine.shutdown()
        self.backend.shutdown()

        # Explicitly finish the logger to prevent hang in __del__ during garbage collection
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.finish()

    # =========================================================================
    # Helper functions
    # =========================================================================
    @staticmethod
    def _collect_workflow_metrics_from_episodes(episodes: list[Episode]) -> tuple[dict, Counter]:
        workflow_metrics = defaultdict(list)
        termination_counts = Counter()
        for episode in episodes:
            for k, v in episode.metrics.items():
                workflow_metrics[k].append(v)
            reason = episode.termination_reason or TerminationReason.UNKNOWN
            termination_counts[getattr(reason, "value", reason)] += 1
        # reduce the metrics to a scalar value, with error handling
        reduced_workflow_metrics = {}
        for k, v in workflow_metrics.items():
            try:
                reduced_workflow_metrics[k] = np.mean(v)
            except Exception:
                continue
        return reduced_workflow_metrics, termination_counts


class TrainerLauncher(ABC):
    """
    A unified agent trainer launcher that directly interfaces with the user script to launch training jobs.

    It handles the necessary environment setup (e.g. ray init for `verl`) for different backends. This is an abstract
    class that each backend must implement.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        store: Store | None = None,
        **kwargs,
    ):
        """Initialize the TrainerLauncher."""
        self.config = config
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.store = store
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwargs = kwargs

    @abstractmethod
    def train(self):
        raise NotImplementedError("Train method of the trainer launcher is not implemented")


class AgentTrainer:
    """
    A unified agent trainer launcher that directly interfaces with the user script to launch training jobs.
    Adapted directly from `rllm.trainer.agent_trainer.AgentTrainer`.

    This trainer will simply delegate the task to the corresponding launcher class.

    Provide exactly one of ``workflow_class`` or (``agent_flow`` AND ``evaluator``).
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend: Literal["verl", "tinker"] = "verl",
        agent_flow: Any = None,
        evaluator: Any = None,
        store: Store | None = None,
        **kwargs,
    ):
        has_agent_flow = agent_flow is not None and evaluator is not None
        remote_runtime_enabled = config.rllm.get("remote_runtime", {}).get("enabled", False)
        if not has_agent_flow and not remote_runtime_enabled:
            assert workflow_class is not None, "Either workflow_class, (agent_flow AND evaluator), or remote_runtime must be provided"

        # Pass agent_flow and evaluator through kwargs for UnifiedTrainer
        if agent_flow is not None:
            kwargs["agent_flow"] = agent_flow
        if evaluator is not None:
            kwargs["evaluator"] = evaluator
        kwargs["backend_name"] = backend

        if backend == "verl":
            from rllm.experimental.verl.verl_launcher import VerlTrainerLauncher

            self.launcher = VerlTrainerLauncher(
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                store=store,
                **kwargs,
            )
        elif backend == "tinker":
            from rllm.trainer.tinker.tinker_launcher import TinkerTrainerLauncher

            self.launcher = TinkerTrainerLauncher(
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                store=store,
                **kwargs,
            )

    def train(self):
        self.launcher.train()
