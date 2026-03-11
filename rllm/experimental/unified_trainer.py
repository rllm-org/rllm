import asyncio
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

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
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
from rllm.experimental.common.visualization import visualize_trajectory_last_steps
from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
from rllm.experimental.episode_buffer import AsyncioEpisodeBuffer, BufferedEpisodeGroup, EpisodeGroupAccumulator
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.utils import EpisodeLogger, Tracking, extract_source_metadata
from rllm.workflows.workflow import TerminationReason, Workflow


@dataclass
class TrainerState:
    """Common trainer state that's backend-agnostic. Reset at each training step."""

    rs_state: RejectionSamplingState = field(default_factory=RejectionSamplingState)
    global_step: int = 0
    epoch: int = 0
    total_steps: int = 0
    is_training: bool = True
    policy_version: int = 0
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
        workflow_class: type[Workflow],
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend_args: dict | None = None,
        *,
        traj_grouping_hook: Callable | None = None,
        traj_group_adv_estimator_map: dict | None = None,
        **kwargs,
    ):
        """Initialize the UnifiedTrainer."""
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # initializing and validating common configs
        self.config = config
        self.rllm_config = config.rllm

        # Read user-defined hooks from kwargs
        self.traj_grouping_hook = traj_grouping_hook or _default_traj_grouping_hook
        # Extract the TrajectoryGroup-specific estimator from kwargs
        self.traj_group_adv_estimator_map = traj_group_adv_estimator_map or {}

        self.backend = backend_cls(config=config, **(backend_args or {}))

        self._validate_and_setup_configs()
        self._setup_logging()

        # Async training config
        async_cfg = self.rllm_config.get("async_training", {})
        self.async_config = AsyncTrainingConfig(
            enabled=async_cfg.get("enabled", False),
            mini_batch_size=async_cfg.get("mini_batch_size", 1),
            streaming_chunks=async_cfg.get("streaming_chunks", 1),
            staleness_threshold=async_cfg.get("staleness_threshold", 0.0),
            trigger_parameter_sync_step=async_cfg.get("trigger_parameter_sync_step", 1),
            partial_rollout=async_cfg.get("partial_rollout", True),
        )

        rollout_engine: RolloutEngine = self.backend.init_rollout_engine(
            cf_config=self.cf_config,
            transform_config=self.transform_config,
            rs_config=self.rs_config,
            algorithm_config=self.algorithm_config,
        )
        self.agent_workflow_engine = UnifiedWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
            retry_limit=self.rllm_config.workflow.retry_limit,
            raise_on_error=self.rllm_config.workflow.raise_on_error,
            episode_logger=self.episode_logger,
        )

        self.tokenizer = None
        if hasattr(self.backend, "tokenizer"):
            self.tokenizer = self.backend.tokenizer

    def _validate_and_setup_configs(self):
        """Validate and setup common configs."""
        # validate common, backend-agnostic configs
        assert self.rllm_config is not None, "rLLM config is not set"
        # if the traj_group_adv_estimator_map is given, the user must turn `use_rllm` to True
        if self.traj_group_adv_estimator_map and not self.rllm_config.algorithm.get("use_rllm", False):
            raise ValueError("If `traj_group_adv_estimator_map` is given, the user must explicitly turn `rllm.algorithm.use_rllm` to True")

        if self.rllm_config.rejection_sample.multiplier != 1:
            assert self.rllm_config.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # validate backend-specific configs
        self.backend.validate_config()

        # compact filtering config (used for filtering out episodes that are not valid)
        self.cf_config = CompactFilteringConfig.from_config(self.rllm_config.compact_filtering)

        # transform config (used for transforming episodes to trajectory groups)
        self.transform_config = TransformConfig(broadcast=self.rllm_config.stepwise_advantage.mode == "broadcast")

        # rejection sampling config (used for rejection sampling)
        rs_mode = "episode" if self.rllm_config.rejection_sample.enable else "none"

        self.rs_config = RejectionSamplingConfig(
            mode=rs_mode,
            min_partial_solve_tasks=self.rllm_config.rejection_sample.min_partial_solve_tasks,
            min_trajs_per_group=self.rllm_config.rejection_sample.min_trajs_per_group,
            filter_uniform_groups=self.rllm_config.rejection_sample.get("filter_uniform_groups", False),
        )

        # algorithm config (used for rLLM-native advantage computation)
        self.algorithm_config = AlgorithmConfig(
            estimator=self.rllm_config.algorithm.adv_estimator,
            estimator_map=self.traj_group_adv_estimator_map,  # TODO(listar2000): see if we can make this configurable in config as well
            stepwise_advantage_mode=self.rllm_config.stepwise_advantage.mode,
            norm_adv_by_std_in_grpo=self.rllm_config.stepwise_advantage.get("norm_adv_by_std_in_grpo", True),
            use_rllm=self.rllm_config.algorithm.get("use_rllm", False),
            use_precomputed_advantage=self.rllm_config.algorithm.get("use_precomputed_advantage", False),
            loss_fn=self.rllm_config.algorithm.get("loss_fn", None),
            lr_schedule=self.rllm_config.algorithm.get("lr_schedule", "constant"),
            warmup_steps_ratio=self.rllm_config.algorithm.get("warmup_steps_ratio", 0.0),
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

    def fit(self):
        """Main training loop (sync entry point)."""
        asyncio.run(self.fit_async())

    async def fit_async(self) -> None:
        """Public async entry point for the full training process."""
        # initialize the UnifiedWorkflowEngine (init the workflow pool)
        await self.agent_workflow_engine.initialize_pool()

        trainer_state = TrainerState()

        await self.backend.on_train_start(trainer_state)

        if self.rllm_config.trainer.get("val_before_train", True):
            val_metrics = await self._validate_async(trainer_state)
            pprint(f"Initial validation metrics: {val_metrics}")
            if self.rllm_config.trainer.get("val_only", False):
                return

        # we start from step (1 + original start batch index)
        trainer_state.global_step += 1

        # Run the training loop
        await self._fit_async(trainer_state)

        await self.backend.on_train_end(trainer_state)

    async def _fit_async(self, trainer_state: TrainerState) -> None:
        """Dispatch to sync or concurrent training based on config."""
        # TODO(listar2000): after some benchmarking, maybe we just keep the fully-async and treat on-policy as a special case.
        if self.async_config.enabled:
            await self._fit_fully_async(trainer_state)
        else:
            await self._fit_on_policy(trainer_state)

    async def _fit_on_policy(self, trainer_state: TrainerState) -> None:
        """Synchronous training loop (the most vanilla, standalone case that does not support minibatching or off-policy training)."""
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
                with simple_timer("total_step", trainer_state.timing_dict):
                    await self._train_batch_async(batch, trainer_state)
                await self.backend.on_batch_end(trainer_state)

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
            val_metrics = await self._validate_async(trainer_state)
            pprint(f"Final validation metrics: {val_metrics}")

    async def _train_batch_async(self, batch: Any, trainer_state: TrainerState) -> None:
        """Train a batch (async implementation)."""
        self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="train", epoch=trainer_state.epoch)

        # stage 1: generate episodes (async) and collect metrics (sync)
        trainer_state.episodes = await self.backend.generate_episodes(batch, agent_workflow_engine=self.agent_workflow_engine, is_validation=False)
        if not trainer_state.has_episodes:
            return

        workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(trainer_state.episodes)

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

        for key, value in workflow_metrics.items():
            trainer_state.metrics[f"batch/{key}"] = np.mean(value)

        total_counts = max(sum(termination_counts.values()), 1)
        for r in TerminationReason:
            trainer_state.metrics[f"batch/termination_reason/{r.value}"] = termination_counts[r.value] / total_counts

    # =========================================================================
    # Fully-asynchronous training pipeline
    # =========================================================================

    async def _fit_fully_async(self, trainer_state: TrainerState) -> None:
        """Fully-async generation + training with group-level streaming."""
        assert self.config.data.train_batch_size == 1, f"Async training requires train_batch_size=1, got {self.config.data.train_batch_size}"
        coord_config = SyncCoordinatorConfig(
            mini_batch_size=self.async_config.mini_batch_size,
            group_size=self.rllm_config.rollout.n,
            staleness_threshold=self.async_config.staleness_threshold,
            trigger_parameter_sync_step=self.async_config.trigger_parameter_sync_step,
        )
        coordinator = SyncCoordinator(coord_config)
        buffer = AsyncioEpisodeBuffer()

        # Compute total_steps for LR scheduling
        train_dataloader = self.backend.get_dataloader(self.train_dataset, trainer_state)
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        if use_total_batches:
            trainer_state.total_steps = self.rllm_config.trainer.total_batches
        else:
            trainer_state.total_steps = len(train_dataloader) * self.rllm_config.trainer.total_epochs

        await asyncio.gather(
            self._generation_loop(trainer_state, buffer, coordinator),
            self._training_loop(trainer_state, buffer, coordinator),
        )

    async def _generation_loop(self, trainer_state: TrainerState, buffer: AsyncioEpisodeBuffer, coordinator: SyncCoordinator) -> None:
        """Generate episodes and stream to buffer. Continuous fire-and-forget per prompt."""
        group_size = self.rllm_config.rollout.n
        accumulator = EpisodeGroupAccumulator(
            group_size=group_size,
            buffer=buffer,
            filter_uniform_groups=self.rs_config.filter_uniform_groups,
            on_group_filtered=coordinator.on_group_filtered,
        )

        try:
            for epoch in range(self.rllm_config.trainer.total_epochs):
                train_dataloader = self.backend.get_dataloader(self.train_dataset, trainer_state)
                self.agent_workflow_engine.set_training_step(trainer_state.global_step, mode="train", epoch=epoch)

                for batch in train_dataloader:
                    # async training uses train_batch_size=1
                    task = batch[0]

                    # Block during validation / non-partial sync
                    await coordinator.wait_for_generation_allowed()

                    # Dispatch-time throttle: block if quota exhausted
                    if not coordinator.has_quota():
                        await coordinator.wait_for_throttle()

                    coordinator.on_group_dispatched()

                    # Generate a unique task_id for this prompt
                    task_id = str(uuid.uuid4())

                    # Fire-and-forget n rollout tasks for this prompt
                    for rollout_idx in range(group_size):

                        async def _run_rollout(t=task, tid=task_id, ridx=rollout_idx):
                            try:
                                _, _, _, episode = await self.agent_workflow_engine.process_task_with_retry(task=t, task_id=tid, rollout_idx=ridx, result_idx=0)
                                await accumulator.add_episode(tid, episode)
                            except Exception:
                                # Group can never complete — free the throttle slot to prevent deadlock
                                coordinator.on_group_consumed()
                                raise

                        asyncio.create_task(_run_rollout())

            # Wait for all in-flight rollouts to finish before marking generation complete
            await self._wait_for_all_workflows_idle()
        finally:
            coordinator.generation_done = True
            buffer.mark_generation_complete()

    async def _training_loop(self, trainer_state: TrainerState, buffer: AsyncioEpisodeBuffer, coordinator: SyncCoordinator) -> None:
        """Collect episode groups from buffer, train with streaming grad accumulation. Runs concurrently with generation."""
        mini_batch_size = self.async_config.mini_batch_size
        streaming_chunks = self.async_config.streaming_chunks
        groups_per_chunk = mini_batch_size // streaming_chunks
        use_total_batches = self.rllm_config.trainer.get("total_batches", -1) > 0
        rollout_engine = self.agent_workflow_engine.rollout_engine

        while True:
            trainer_state.reset_batch()
            step_start = time.perf_counter()
            all_collected: list[BufferedEpisodeGroup] = []
            all_episodes: list[Episode] = []
            buffer_wait_time = 0.0

            # 1. Streaming gradient accumulation across chunks
            for chunk_idx in range(streaming_chunks):
                # Pull groups_per_chunk groups from buffer
                chunk_groups: list[BufferedEpisodeGroup] = []
                while len(chunk_groups) < groups_per_chunk:
                    t0 = time.perf_counter()
                    item = await buffer.get()
                    buffer_wait_time += time.perf_counter() - t0
                    if item is None:
                        break  # generation done + buffer empty
                    chunk_groups.append(item)

                if not chunk_groups:
                    break

                for _ in chunk_groups:
                    coordinator.on_group_consumed()
                all_collected.extend(chunk_groups)

                # Flatten episodes from groups
                episodes = []
                for group in chunk_groups:
                    episodes.extend(group.episodes)
                all_episodes.extend(episodes)

                # Transform → rejection sampling → backend pipeline
                trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(
                    episodes,
                    self.transform_config,
                    self.cf_config,
                    traj_grouping_hook=self.traj_grouping_hook,
                )
                trainer_state.trajectory_groups = trajectory_groups
                trainer_state.episodes = episodes
                trainer_state.metrics.update(transform_metrics)

                filtered_groups, filtered_episodes, rs_metrics = apply_rejection_sampling_and_filtering(
                    episodes,
                    trajectory_groups,
                    self.rs_config,
                    RejectionSamplingState(),
                )
                trainer_state.metrics.update(rs_metrics)
                trainer_state.trajectory_groups = filtered_groups
                trainer_state.episodes = filtered_episodes
                if not trainer_state.has_trajectory_groups:
                    continue

                await self.backend.on_batch_start(trainer_state)
                trainer_state.backend_batch = self.backend.transform_to_backend_batch(trainer_state)
                await self.backend.process_backend_batch(trainer_state)
                await self.backend.compute_advantages(trainer_state, self.algorithm_config)

            if not all_collected:
                if coordinator.generation_done and buffer.qsize() == 0:
                    break
                continue

            # 2. Single optimizer step
            trainer_state.episodes = all_episodes
            await self.backend.update_policy(trainer_state)

            # 3. Training step done — check sync
            coordinator.on_training_step_complete()
            sync_time = 0.0
            if coordinator.should_sync():
                t0 = time.perf_counter()
                await self._perform_weight_sync(trainer_state, coordinator, rollout_engine)
                sync_time = time.perf_counter() - t0

            # 4. Metrics, logging, visualization
            workflow_metrics, termination_counts = self._collect_workflow_metrics_from_episodes(all_episodes)

            staleness_values = [coordinator.policy_version - g.weight_version for g in all_collected]
            trainer_state.metrics["async/staleness_mean"] = np.mean(staleness_values)
            trainer_state.metrics["async/staleness_min"] = np.min(staleness_values)
            trainer_state.metrics["async/staleness_max"] = np.max(staleness_values)
            trainer_state.metrics["async/groups_consumed"] = len(all_collected)

            # Timing
            trainer_state.metrics["time/step"] = time.perf_counter() - step_start
            trainer_state.metrics["time/buffer_wait"] = buffer_wait_time
            if sync_time > 0:
                trainer_state.metrics["time/weight_sync"] = sync_time

            # Weight version delta within trajectories (meaningful in partial_rollout mode)
            traj_deltas = []
            for ep in all_episodes:
                for traj in ep.trajectories:
                    versions = [s.weight_version for s in traj.steps if s.weight_version is not None]
                    if len(versions) >= 2:
                        traj_deltas.append(max(versions) - min(versions))
            if traj_deltas:
                trainer_state.metrics["async/traj_weight_delta_mean"] = np.mean(traj_deltas)
                trainer_state.metrics["async/traj_weight_delta_min"] = np.min(traj_deltas)
                trainer_state.metrics["async/traj_weight_delta_max"] = np.max(traj_deltas)

            buffer_stats = buffer.stats()
            trainer_state.metrics["async/gen_train_ratio"] = buffer_stats["async/total_produced"] / max(trainer_state.global_step, 1)
            trainer_state.metrics.update(buffer_stats)
            trainer_state.metrics.update(coordinator.stats())

            if self.tokenizer is not None and trainer_state.has_trajectory_groups:
                visualize_trajectory_last_steps(
                    trainer_state.trajectory_groups,
                    tokenizer=self.tokenizer,
                    max_steps_to_visualize=2,
                    show_workflow_metadata=True,
                )

            for key, value in workflow_metrics.items():
                trainer_state.metrics[f"batch/{key}"] = np.mean(value)

            total_counts = max(sum(termination_counts.values()), 1)
            for r in TerminationReason:
                trainer_state.metrics[f"batch/termination_reason/{r.value}"] = termination_counts[r.value] / total_counts

            await self.backend.on_batch_end(trainer_state)

            self.logger.log(
                data=trainer_state.metrics,
                step=trainer_state.global_step,
                episodes=all_episodes,
                trajectory_groups=trainer_state.trajectory_groups,
            )

            # Periodic validation
            if self.rllm_config.trainer.test_freq > 0 and trainer_state.global_step % self.rllm_config.trainer.test_freq == 0:
                await self._validate_async_with_pause(trainer_state, coordinator)

            trainer_state.global_step += 1

            # Check total_batches limit
            if use_total_batches and trainer_state.global_step >= self.rllm_config.trainer.total_batches:
                break

    async def _perform_weight_sync(self, trainer_state: TrainerState, coordinator: SyncCoordinator, rollout_engine: RolloutEngine) -> None:
        """Synchronize weights between training and rollout engines.

        Two modes depending on partial_rollout:
        - partial_rollout=True: Uses rollout engine gate (model-call level).
          Workflows block between turns, resume with new weights.
        - partial_rollout=False: Uses coordinator generation pause (dispatch level).
          Workflows finish naturally, gate stays open.
        """
        if self.async_config.partial_rollout:
            # Block new model calls; in-flight calls finish, workflows pause between turns
            rollout_engine.close_gate()
            await rollout_engine.wait_for_drain()
        else:
            # Stop dispatching new prompts, let all workflows finish naturally
            coordinator.pause_generation()
            await self._wait_for_all_workflows_idle()

        trainer_state.policy_version = coordinator.policy_version + 1
        await self.backend.on_policy_updated(trainer_state)
        rollout_engine.weight_version = trainer_state.policy_version
        coordinator.on_sync_complete()

        if self.async_config.partial_rollout:
            rollout_engine.open_gate()
        else:
            coordinator.resume_generation()

    async def _wait_for_all_workflows_idle(self) -> None:
        """Wait for all n_parallel_tasks workflows to return to the pool."""
        pool = self.agent_workflow_engine
        while pool.workflow_queue.qsize() < pool.n_parallel_tasks:
            await asyncio.sleep(0.1)

    async def _validate_async_with_pause(self, trainer_state: TrainerState, coordinator: SyncCoordinator) -> dict:
        """Validation with dispatch-level pause. Waits for workflows to drain, then runs validation."""
        coordinator.pause_generation()
        await self._wait_for_all_workflows_idle()
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
            val_trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(val_episodes, self.transform_config, self.cf_config, traj_grouping_hook=self.traj_grouping_hook)
            reward_metrics = collect_reward_and_advantage_from_trajectory_groups(val_trajectory_groups, self.algorithm_config, collect_advantage=False)

            is_correct_lst.extend([episode.is_correct for episode in val_episodes])
            uid_lst.extend([episode.task_id for episode in val_episodes])

            data_sources = [episode.info.get("data_source", "unknown") for episode in val_episodes]
            data_source_lst.extend(data_sources)

            for episode, data_source in zip(val_episodes, data_sources, strict=True):
                for key, value in episode.metrics.items():
                    workflow_metrics_by_source[data_source][key].append(float(value))

            for key, value in (transform_metrics | reward_metrics).items():
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
        self.logger.log(data=val_metrics, step=trainer_state.global_step)
        await self.backend.on_validation_end(trainer_state)
        return val_metrics

    def shutdown(self):
        """Shutdown the trainer and cleanup resources."""
        if hasattr(self, "agent_workflow_engine") and self.agent_workflow_engine is not None:
            self.agent_workflow_engine.shutdown()
        self.backend.shutdown()

        # Explicitly finish the logger to prevent hang in __del__ during garbage collection
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.finish()

    # =========================================================================
    # Helper functions
    # =========================================================================
    def _collect_workflow_metrics_from_episodes(self, episodes: list[Episode]) -> tuple[dict, Counter]:
        workflow_metrics = defaultdict(list)
        termination_counts = Counter()
        for episode in episodes:
            for k, v in episode.metrics.items():
                workflow_metrics[k].append(v)
            if episode.termination_reason is not None:
                termination_counts[episode.termination_reason.value] += 1
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

    TODO(listar2000): add support to non-workflow training (e.g. agent/env classes), `fireworks` backend, and SDK.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow],
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the TrainerLauncher."""
        self.config = config
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
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
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow],
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend: Literal["verl", "tinker", "fireworks"] = "verl",
        **kwargs,
    ):
        match backend:
            case "verl":
                from rllm.experimental.verl.verl_launcher import VerlTrainerLauncher

                launcher_cls = VerlTrainerLauncher
            case "tinker":
                from rllm.trainer.tinker.tinker_launcher import TinkerTrainerLauncher

                launcher_cls = TinkerTrainerLauncher
            case "fireworks":
                from rllm.trainer.fireworks.fireworks_launcher import FireworksTrainerLauncher

                launcher_cls = FireworksTrainerLauncher
            case _:
                raise ValueError(f"Unsupported backend: {backend}, must be one of ['verl', 'tinker', 'fireworks']")

        self.launcher = launcher_cls(
            config=config,
            workflow_class=workflow_class,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            workflow_args=workflow_args,
            **kwargs,
        )

    def train(self):
        self.launcher.train()
