"""
Tinker backend implementation for the UnifiedTrainer.

This backend implements the BackendProtocol interface to provide
Tinker-specific implementations for the unified training pipeline.

TODO(listar2000): in the future, when implementing PPO mini-batching for Tinker, we should be careful
about the update of self.sampling_client (currently update only once per batch).
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

import tinker
from rllm.agents.agent import Episode
from rllm.data import Dataset
from rllm.experimental.common import AlgorithmConfig, simple_timer
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.rollout import RolloutEngine, TinkerEngine
from rllm.experimental.tinker.tinker_metrics_utils import (
    print_metrics_table,
    update_training_metrics,
)
from rllm.experimental.tinker.tinker_policy_trainer import TinkerPolicyTrainer

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


def _build_interleave_batch(batch: list[dict], group_size: int) -> list[dict]:
    """Build an interleaved batch where each task is repeated `group_size` times."""
    interleave_batch = []
    batch_with_uid = []
    for batch_item in batch:
        batch_with_uid.append({**batch_item, "uid": str(uuid.uuid4())})

    for batch_item in batch_with_uid:
        interleave_batch.extend([batch_item for _ in range(group_size)])
    return interleave_batch


class TinkerBackend(BackendProtocol[Iterable, list[tinker.Datum]]):
    """
    Tinker backend for the unified trainer.

    This backend provides:
        - Tinker-specific rollout engine (TinkerEngine)
        - Policy training via TinkerPolicyTrainer
        - Checkpoint management via Tinker's checkpoint utilities

    The backend uses async methods naturally to match Tinker's async API.
    """

    name: str = "tinker"
    requires_loop: bool = True  # Tinker uses async operations

    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        """Initialize the TinkerBackend.

        Args:
            config: The full configuration object.
            **kwargs: Additional arguments.
        """
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference
        self.full_config = config

        # Tinker service client
        self.service_client = tinker.ServiceClient(base_url=config.tinker_base_url)

        # Initialize policy trainer (filled during init_rollout_engine)
        self.policy_trainer: TinkerPolicyTrainer | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.learning_rate = self.full_config.training.learning_rate

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: TinkerEngine | None = None

        # Sampling client - updated after each checkpoint save
        self.sampling_client: tinker.SamplingClient | None = None
        # Store algorithm config for use in process_backend_batch
        self._algorithm_config: AlgorithmConfig | None = None

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the TinkerEngine rollout engine.

        Args:
            **kwargs: Additional arguments, including the various configurations

        Returns:
            TinkerEngine: The initialized rollout engine.
        """
        self.policy_trainer = TinkerPolicyTrainer(
            config=self.full_config,
            service_client=self.service_client,
            cf_config=kwargs.get("cf_config"),
            transform_config=kwargs.get("transform_config"),
            algorithm_config=kwargs.get("algorithm_config"),
        )
        # we need to get it from `AutoTokenizer` since the `policy_trainer` has not been initialized yet
        self.tokenizer = AutoTokenizer.from_pretrained(self.full_config.model.name)

        self.rollout_engine = TinkerEngine(
            base_url=self.full_config.tinker_base_url,
            model_name=self.full_config.model.name,
            service_client=self.service_client,
            tokenizer=self.tokenizer,
            max_prompt_length=self.full_config.data.max_prompt_length,
            max_response_length=self.full_config.data.max_response_length,
            sampling_params=self.full_config.sampling,
        )
        return self.rollout_engine

    def validate_config(self) -> None:
        """Validate Tinker-specific configuration settings."""
        # Check for recommended sampling parameters
        sampling_params = self.full_config.sampling
        if sampling_params.get("temperature", 1.0) != 1.0 or sampling_params.get("top_p", 1.0) != 1.0:
            logger.warning("Temperature and top_p are set away from 1.0, this is not recommended by Tinker and can cause mysterious issues with logprobs. See https://github.com/thinking-machines-lab/tinker-cookbook/pull/86 for discussion.")

        # Validate num_minibatches (currently only support 1)
        if self.full_config.training.get("num_minibatches", 1) != 1:
            logger.warning(f"Only num_minibatches=1 is fully tested for TinkerBackend, current num_minibatches={self.full_config.training.num_minibatches}")

    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> Iterable:
        """Get dataloader for the given dataset.

        For Tinker, we create standard PyTorch DataLoaders.

        Args:
            dataset: The dataset to create dataloader from.
            trainer_state: The trainer state.

        Returns:
            DataLoader wrapped dataset.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None for TinkerBackend")

        if trainer_state.is_training:
            batch_size = self.full_config.data.train_batch_size
            shuffle = True
        else:
            batch_size = self.full_config.data.get("val_batch_size", self.full_config.data.train_batch_size)
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: x,  # Return batches as lists
        )

    def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        super().shutdown()

    # =========================================================================
    # Async pipeline methods
    # =========================================================================

    async def generate_episodes(
        self,
        batch: Any,
        agent_workflow_engine: UnifiedWorkflowEngine,
        **kwargs,
    ) -> list[Episode]:
        """Generate episodes using the workflow engine.

        For Tinker backend, this function handles:
        1. Building an interleaved batch (each task repeated `group_size` times)
        2. Setting the sampling client on the rollout engine
        3. Executing tasks using the agent workflow engine

        Args:
            batch: Input batch (list of task dicts from dataloader).
            agent_workflow_engine: The workflow engine to use for episode generation.
            **kwargs: Additional arguments.

        Returns:
            List of generated episodes.
        """
        assert self.rollout_engine is not None, "rollout_engine is not initialized"
        assert self.sampling_client is not None, "sampling_client is not initialized"

        # Set the sampling client on the rollout engine
        self.rollout_engine.set_sampling_client(self.sampling_client)

        # Build interleaved batch
        if agent_workflow_engine.current_mode == "train":
            group_size = self.full_config.rllm.rollout.n
        else:
            group_size = self.full_config.rllm.rollout.n_val
        interleaved_batch = _build_interleave_batch(batch, group_size)

        # Extract task IDs
        task_ids = [item["uid"] for item in interleaved_batch]

        # Execute tasks using the agent workflow engine (async)
        episodes = await agent_workflow_engine.execute_tasks(interleaved_batch, task_ids, **kwargs)

        return episodes

    def transform_to_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> list[tinker.Datum]:
        """Transform rllm-native data structures to Tinker Datum format.

        Note: For Tinker, the actual transformation and advantage computation
        is done in process_backend_batch via TinkerPolicyTrainer.
        This method returns an empty placeholder.

        Args:
            trainer_state: The trainer state containing rllm-native data structures.
            **kwargs: Additional arguments.

        Returns:
            Empty list (placeholder - actual datums created in process_backend_batch).
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        # Return empty list as placeholder; actual datums are created in process_backend_batch
        return []

    async def process_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Process the backend batch by running forward-backward pass.

        For Tinker, this performs:
        1. Transform trajectory groups to datums (includes advantage computation)
        2. Run forward-backward pass on the training client
        3. Store logprobs for KL metrics computation

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        assert self.policy_trainer is not None, "policy_trainer is not initialized"
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"

        # Use TinkerPolicyTrainer's method for forward-backward
        with simple_timer("forward_backward", trainer_state.timing_dict):
            (
                training_datums,
                training_logprobs,
                adv_metrics,
            ) = await self.policy_trainer.forward_backward_from_trajectory_groups(
                trainer_state.trajectory_groups,
                algorithm_config=self._algorithm_config,
            )

        # Store datums as backend batch
        trainer_state.backend_batch = training_datums
        # Also store the training logprobs
        trainer_state.extra_info["training_logprobs"] = training_logprobs
        # Also store the advantage metrics
        trainer_state.metrics.update(adv_metrics)

    async def compute_advantages(
        self,
        trainer_state: TrainerState,
        algorithm_config: AlgorithmConfig,
        **kwargs,
    ) -> None:
        """Compute advantages from trajectory groups.

        For Tinker, advantage computation is done in process_backend_batch via
        transform_trajectory_groups_to_datums. This method stores the algorithm
        config for use in process_backend_batch.

        Note: This is called BEFORE process_backend_batch in the pipeline,
        so we just store the config here.
        """
        # Store algorithm config for use in process_backend_batch
        self._algorithm_config = algorithm_config

    async def update_policy(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Update the policy via optimizer step.

        For Tinker, this performs the optimizer step after forward-backward.

        Args:
            trainer_state: The trainer state.
        """
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        beta1 = self.full_config.training.get("beta1", 0.9)
        beta2 = self.full_config.training.get("beta2", 0.95)
        eps = self.full_config.training.get("eps", 1e-8)

        # Optimizer step (async)
        with simple_timer("optim_step", trainer_state.timing_dict):
            optim_step_future = await self.policy_trainer.optim_step_future(
                learning_rate=self.learning_rate,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )
            await optim_step_future.result_async()

    # =========================================================================
    # Async hook methods
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        Initializes the policy trainer and loads checkpoints if available.
        """
        assert self.policy_trainer is not None, "policy_trainer is not initialized"
        # Ensure checkpoint directory exists
        os.makedirs(self.full_config.training.default_local_dir, exist_ok=True)

        # Initialize training client and load checkpoint
        start_batch, self.sampling_client = await self.policy_trainer.initialize_async(resume_from_checkpoint=True)

        # Update trainer state with the start batch from checkpoint
        trainer_state.global_step = start_batch

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of training."""
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        # Save final checkpoint if we didn't just save it in the last batch
        if trainer_state.global_step % self.full_config.rllm.trainer.save_freq != 0:
            logger.info(f"Saving final checkpoint at step {trainer_state.global_step}")
            await self.policy_trainer.save_checkpoint_async(trainer_state.global_step, kind="state")

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch.

        Saves checkpoint, updates sampling client, and prints metrics.
        """
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        global_step = trainer_state.global_step
        # Save sampler checkpoint after each batch
        with simple_timer("save_checkpoint", trainer_state.timing_dict):
            logger.info(f"Saving state checkpoint and sampler at step {global_step}")
            save_freq = self.full_config.rllm.trainer.save_freq
            do_save = save_freq > 0 and global_step % save_freq == 0
            self.sampling_client = await self.policy_trainer.save_checkpoint_and_get_sampling_client(global_step, kind="both", do_save=do_save)

        # Update metrics
        total_batches = self.full_config.rllm.trainer.get("total_batches", None)
        update_training_metrics(trainer_state, self.learning_rate, total_batches)

        # Update metrics
        total_batches = self.full_config.rllm.trainer.get("total_batches", None)
        update_training_metrics(trainer_state, self.learning_rate, total_batches)

        # Print metrics table
        if trainer_state.metrics:
            print_metrics_table(trainer_state.metrics, global_step)

    async def on_epoch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of an epoch."""
        logger.info(f"Starting epoch {trainer_state.epoch}")

    async def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of an epoch."""
        logger.info(f"Completed epoch {trainer_state.epoch}")

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation.

        Returns:
            bool: True if validation should proceed, False otherwise.
        """
        trainer_state.is_training = False
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
