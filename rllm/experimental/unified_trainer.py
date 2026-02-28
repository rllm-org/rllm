import asyncio
import time
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
from rllm.experimental.common.advantage import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.experimental.common.config import (
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
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.experimental.common.visualization import visualize_trajectory_last_steps
from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.rollout import RolloutEngine
from rllm.utils import EpisodeLogger, Tracking
from rllm.workflows.workflow import TerminationReason, Workflow


@dataclass
class TrainerState:
    """Common trainer state that's backend-agnostic. Reset at each training step."""

    rs_state: RejectionSamplingState = field(default_factory=RejectionSamplingState)
    global_step: int = 0
    epoch: int = 0
    total_steps: int = 0
    is_training: bool = True
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
        agent_run_func: Callable | None = None,
        **kwargs,
    ):
        """Initialize the UnifiedTrainer.

        Provide exactly one of ``workflow_class`` or ``agent_run_func``.
        When ``agent_run_func`` is given, the trainer automatically creates an
        ``SdkWorkflow`` adapter so that SDK users can leverage the unified
        pipeline (rejection sampling, advantage computation, verl + tinker).

        When ``sandbox.enabled`` is true in the SDK config, ``agent_run_func``
        may be ``None`` — the sandbox orchestrator handles agent execution.
        """
        sandbox_enabled = config.rllm.get("sdk", {}).get("sandbox", {}).get("enabled", False)
        if not sandbox_enabled:
            assert (workflow_class is not None) ^ (agent_run_func is not None), "Exactly one of workflow_class or agent_run_func must be provided"

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # initializing and validating common configs
        self.config = config
        self.rllm_config = config.rllm

        # Read user-defined hooks from kwargs
        self.trajectory_grouping_hook = kwargs.get("trajectory_grouping_hook")

        # SDK factory (set up later if agent_run_func is provided)
        self._sdk_factory = None

        self.backend = backend_cls(config=config, **(backend_args or {}))

        self._validate_and_setup_configs()
        self._setup_logging()

        rollout_engine: RolloutEngine = self.backend.init_rollout_engine(
            cf_config=self.cf_config,
            transform_config=self.transform_config,
            rs_config=self.rs_config,
            algorithm_config=self.algorithm_config,
        )

        # If agent_run_func is provided (or sandbox enabled), set up SDK workflow via factory
        post_execute_hook = None
        if agent_run_func is not None or sandbox_enabled:
            from rllm.experimental.engine.sdk_workflow import SdkWorkflow, SdkWorkflowFactory

            self._sdk_factory = SdkWorkflowFactory(
                agent_run_func=agent_run_func,
                rollout_engine=rollout_engine,
                config=self.config,
            )
            self.workflow_class = SdkWorkflow
            self.workflow_args = self._sdk_factory.get_workflow_args()
            post_execute_hook = self._sdk_factory.flush_traces_hook

        self.agent_workflow_engine = UnifiedWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.rllm_config.workflow.n_parallel_tasks,
            retry_limit=self.rllm_config.workflow.retry_limit,
            raise_on_error=self.rllm_config.workflow.raise_on_error,
            episode_logger=self.episode_logger,
            post_execute_hook=post_execute_hook,
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

        # TODO(listar2000): add these configurations to the hydra config
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
        )

        # algorithm config (used for rLLM-native advantage computation)
        self.algorithm_config = AlgorithmConfig(
            estimator=self.rllm_config.algorithm.adv_estimator,
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

        self.logger = Tracking(
            project_name=self.rllm_config.trainer.project_name,
            experiment_name=self.rllm_config.trainer.experiment_name,
            default_backend=self.rllm_config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    # =========================================================================
    # Main training loop methods
    # =========================================================================

    def fit(self):
        """Main training loop (sync entry point)."""
        asyncio.run(self.fit_async())

    async def fit_async(self) -> None:
        """Public async entry point for the full training process."""
        # Initialize sandbox orchestrator (if enabled) before the workflow pool
        if self._sdk_factory is not None:
            await self._sdk_factory.initialize_sandbox()

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
        """Internal async main training loop."""
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

                self.logger.log(data=trainer_state.metrics, step=trainer_state.global_step)

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
        trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(trainer_state.episodes, self.transform_config, self.trajectory_grouping_hook)
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
            val_trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(val_episodes, self.transform_config, self.trajectory_grouping_hook)
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
        if hasattr(self, "_sdk_factory") and self._sdk_factory is not None:
            self._sdk_factory.shutdown()
            self._sdk_factory = None
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
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        agent_run_func: Callable | None = None,
        **kwargs,
    ):
        """Initialize the TrainerLauncher."""
        self.config = config
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.agent_run_func = agent_run_func
        self.kwargs = kwargs

    @abstractmethod
    def train(self):
        raise NotImplementedError("Train method of the trainer launcher is not implemented")


class AgentTrainer:
    """
    A unified agent trainer launcher that directly interfaces with the user script to launch training jobs.
    Adapted directly from `rllm.trainer.agent_trainer.AgentTrainer`.

    This trainer will simply delegate the task to the corresponding launcher class.

    Provide exactly one of ``workflow_class`` or ``agent_run_func``.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        backend: Literal["verl", "tinker"] = "verl",
        agent_run_func: Callable | None = None,
        **kwargs,
    ):
        sandbox_enabled = config.rllm.get("sdk", {}).get("sandbox", {}).get("enabled", False)
        if not sandbox_enabled:
            assert (workflow_class is not None) ^ (agent_run_func is not None), "Exactly one of workflow_class or agent_run_func must be provided"

        if backend == "verl":
            from rllm.experimental.verl.verl_launcher import VerlTrainerLauncher

            self.launcher = VerlTrainerLauncher(
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                agent_run_func=agent_run_func,
                **kwargs,
            )
        elif backend == "tinker":
            from rllm.experimental.tinker.tinker_launcher import TinkerTrainerLauncher

            self.launcher = TinkerTrainerLauncher(
                config=config,
                workflow_class=workflow_class,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                workflow_args=workflow_args,
                agent_run_func=agent_run_func,
                **kwargs,
            )

    def train(self):
        self.launcher.train()
