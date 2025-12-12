import asyncio
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.engine.unified_workflow_engine import UnifiedWorkflowEngine
from rllm.experimental.base import BackendProtocol
from rllm.trainer.common import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.common.rejection_sampling import RejectionSamplingState, apply_rejection_sampling_and_filtering
from rllm.trainer.common.transform import transform_episodes_to_trajectory_groups
from rllm.utils import EpisodeLogger, Tracking, visualize_trajectory_last_steps
from rllm.workflows.workflow import TerminationReason, Workflow


@dataclass
class TrainerState:
    """Common trainer state that's backend-agnostic. Reset at each training step."""

    rs_state: RejectionSamplingState = field(default_factory=RejectionSamplingState)
    global_step: int = 0
    epoch: int = 0
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
        return self.episodes is not None

    @property
    def has_trajectory_groups(self) -> bool:
        return self.trajectory_groups is not None

    @property
    def has_backend_batch(self) -> bool:
        return self.backend_batch is not None


class UnifiedTrainer:
    """Unified trainer for backend-agnostic training."""

    def __init__(self, backend_cls: type[BackendProtocol], config: DictConfig, workflow_class: type[Workflow], train_dataset: Dataset, val_dataset: Dataset | None = None, workflow_args: dict | None = None, backend_args: dict | None = None, **kwargs):
        """
        Initialize the UnifiedTrainer.
        """
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.backend = backend_cls(config=self.backend_config, **(backend_args or {}))

        self.tokenizer = kwargs.get("tokenizer")

        # initializing and validating common configs
        self.config = config
        self.backend_config = config.get(backend_cls.name, DictConfig({}))

        self._validate_and_setup_configs()
        self._setup_event_loop()  # set up event loop for both agent workflow engine and (optionally) backend

        # create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        rollout_engine: RolloutEngine = self.backend.init_rollout_engine()  # obtain rollout engine from backend
        self.agent_workflow_engine = UnifiedWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            raise_on_error=self.config.rllm.workflow.raise_on_error,
            episode_logger=episode_logger,
        )

        asyncio.run_coroutine_threadsafe(self.agent_workflow_engine.initialize_pool(), self._loop).result()  # type: ignore

    def _validate_and_setup_configs(self):
        """Validate and setup common configs."""
        # validate common, backend-agnostic configs
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert self.config.rllm.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # validate backend-specific configs
        self.backend.validate_config()

        # TODO(listar2000): add these configurations to the hydra config
        # compact filtering config (used for filtering out episodes that are not valid)
        self.cf_config = CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)

        # transform config (used for transforming episodes to trajectory groups)
        self.transform_config = TransformConfig(broadcast=self.config.rllm.stepwise_advantage.mode == "broadcast")

        # rejection sampling config (used for rejection sampling)
        rs_mode = "episode" if self.config.rllm.rejection_sample.enable else "none"

        self.rs_config = RejectionSamplingConfig(
            mode=rs_mode,
            min_partial_solve_tasks=self.config.rllm.rejection_sample.min_partial_solve_tasks,
            min_trajs_per_group=self.config.rllm.rejection_sample.min_trajs_per_group,
        )

        # algorithm config (used for rLLM-native advantage computation)
        self.algorithm_config = AlgorithmConfig(
            estimator=self.config.algorithm.adv_estimator,
            stepwise_advantage_mode=self.config.rllm.stepwise_advantage.mode,
            norm_adv_by_std_in_grpo=self.config.rllm.stepwise_advantage.get("norm_adv_by_std_in_grpo", True),
        )

    def _setup_event_loop(self):
        """Setup the event loop for the backend. Only invoke this if the backend requires a loop."""
        import threading

        assert self._loop is None and self._thread is None, "Event loop already set up. _setup_event_loop should not be called twice."

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    # =========================================================================
    # Main training loop methods
    # =========================================================================
    def fit(self):
        """Main training loop."""
        trainer_state = TrainerState()
        self.backend.on_train_start(trainer_state)

        if self.config.trainer.get("val_before_train", True):
            self.validate(trainer_state)
            if self.config.trainer.get("val_only", False):
                return

        trainer_state.global_step += 1  # we start from step 1
        # (optionally) convert the train dataset to backend-specific format
        train_dataloader: Iterable = self.backend.get_dataloader(self.train_dataset)

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {trainer_state.global_step} started")
            self.backend.on_epoch_start(trainer_state)
            for batch in train_dataloader:
                trainer_state.reset_batch()
                # optional pre-batch hook: manage things like start profiling, etc.
                self.backend.on_batch_start(trainer_state)
                self._train_batch(batch, trainer_state)
                # optional post-batch hook
                self.backend.on_batch_end(trainer_state)

                trainer_state.global_step += 1

                # periodic validation
                if self.config.trainer.get("val_freq", 0) > 0 and trainer_state.global_step % self.config.trainer.val_freq == 0:
                    self.validate(trainer_state)

            self.backend.on_epoch_end(trainer_state)
            trainer_state.epoch += 1

    def _train_batch(self, batch: Any, trainer_state: TrainerState) -> None:
        """Train a batch."""
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)

        # stage 1: generate episodes
        trainer_state.episodes = self.backend.generate_episodes(batch, loop=self._loop)

        # stage 2: transform episodes to trajectory groups
        trajectory_groups, transform_metrics = transform_episodes_to_trajectory_groups(trainer_state.episodes, self.transform_config)
        trainer_state.trajectory_groups = trajectory_groups
        trainer_state.metrics.update(transform_metrics)

        # stage 3: apply rejection sampling
        filtered_groups, filtered_episodes, rs_metrics = apply_rejection_sampling_and_filtering(trainer_state.episodes, trainer_state.trajectory_groups, self.rs_config, trainer_state.rs_state)
        trainer_state.metrics.update(rs_metrics)
        trainer_state.trajectory_groups = filtered_groups
        trainer_state.episodes = filtered_episodes

        if len(filtered_groups) == 0:
            return

        # stage 4: transform trajectory groups to backend-specific format
        backend_batch = self.backend.transform_trajectory_groups_to_backend_batch(trainer_state.trajectory_groups)

        # stage 5: we perform some backend-specific operations, such as computing reference log probs, critic values, etc.
        trainer_state.backend_batch = self.backend.process_backend_batch(backend_batch)

        # stage 6: compute advantages from trajectory groups and update them into the backend batch
        self.backend.compute_advantages(trainer_state.trajectory_groups, trainer_state.backend_batch, self.algorithm_config)
        assert trainer_state.trajectory_groups[0].trajectories[0].steps[0].advantage is not None, "Advantage is not computed"

        # stage 7: backend will update the policy
        self.backend.update_policy(trainer_state.backend_batch)

        # stage 8: cleanup, logging, visualization, etc.
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
            trainer_state.metrics[f"batch/{r.value}"] = termination_counts[r.value] / total_counts

        self.logger.log(data=trainer_state.metrics, step=trainer_state.global_step)

    def validate(self, trainer_state: TrainerState) -> None:
        """Validate the model."""
        if self.val_dataset is None:
            return

        n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n

        self.backend.on_validation_start(trainer_state)

        # TODO(listar2000): check whether we need to log by "source"
        is_correct_lst, uid_lst, data_source_lst = [], [], []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

        val_dataloader: Iterable = self.backend.get_dataloader(self.val_dataset)
        for batch in val_dataloader:
            self.backend.on_batch_start(trainer_state)
            val_episodes = self.backend.generate_episodes(batch, loop=self._loop)
            is_correct_lst.extend([episode.is_correct for episode in val_episodes])
            uid_lst.extend([episode.id.split(":")[0] for episode in val_episodes])
            data_source_lst.extend([episode.info.get("data_source", "unknown") for episode in val_episodes])
            self.backend.on_batch_end(trainer_state)

            for episode, data_source in zip(val_episodes, data_source_lst, strict=True):
                for key, value in episode.metrics.items():
                    workflow_metrics_by_source[data_source][key].append(float(value))

        val_metrics = {}
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
                    if values:  # Only add if we have values
                        val_metrics[f"val/{data_source}/{key}"] = np.mean(values)

        self.logger.log(data=val_metrics, step=trainer_state.global_step)
        self.backend.on_validation_end(trainer_state)

    def shutdown(self):
        """Shutdown the trainer and cleanup resources."""
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()
        if hasattr(self, "agent_workflow_engine") and self.agent_workflow_engine is not None:
            self.agent_workflow_engine.shutdown()
        self.backend.shutdown()
