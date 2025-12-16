"""
Tinker backend implementation for the UnifiedTrainer.

This backend implements the BackendProtocol interface to provide
Tinker-specific implementations for the unified training pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig

import tinker
from rllm.agents.agent import Episode
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.experimental.common.advantage import AlgorithmConfig
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.tinker.tinker_metrics_utils import compute_kl_and_entropy_metrics, print_metrics_table
from rllm.experimental.tinker.tinker_policy_trainer import TinkerPolicyTrainer

if TYPE_CHECKING:
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
        # Initialize BackendProtocol
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference
        self.full_config = config

        # Tinker service client
        self.service_client = tinker.ServiceClient(base_url=config.tinker_base_url)

        # Policy trainer - handles gradient updates and checkpointing
        self.policy_trainer: TinkerPolicyTrainer | None = None

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: TinkerEngine | None = None

        # Sampling client - updated after each checkpoint save
        self.sampling_client: tinker.SamplingClient | None = None

        # Training state
        self._start_batch: int = 0

        # Store training datums and logprobs for KL metrics computation
        self._training_datums: list[tinker.Datum] = []
        self._training_logprobs: list[torch.Tensor] = []

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self) -> RolloutEngine:
        """Initialize the TinkerEngine rollout engine.

        Returns:
            TinkerEngine: The initialized rollout engine.
        """
        self.rollout_engine = TinkerEngine(
            base_url=self.full_config.tinker_base_url,
            model_name=self.full_config.model.name,
            service_client=self.service_client,
            tokenizer=self.service_client.get_tokenizer(self.full_config.model.name),
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
        # Tinker cleanup is handled automatically
        pass

    def generate_episodes(
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
            **kwargs: Additional arguments including 'loop' for async execution.

        Returns:
            List of generated episodes.
        """
        assert "loop" in kwargs and kwargs["loop"] is not None, "async event loop is required"
        loop = kwargs["loop"]

        assert self.rollout_engine is not None, "rollout_engine is not initialized"
        assert self.sampling_client is not None, "sampling_client is not initialized"

        # Set the sampling client on the rollout engine
        self.rollout_engine.set_sampling_client(self.sampling_client)

        # Build interleaved batch
        group_size = self.full_config.training.group_size
        interleaved_batch = _build_interleave_batch(batch, group_size)

        # Extract task IDs
        task_ids = [item["uid"] for item in interleaved_batch]

        # Execute tasks using the agent workflow engine
        coro = agent_workflow_engine.execute_tasks(interleaved_batch, task_ids, **kwargs)

        if loop is not None:
            episodes = asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            episodes = asyncio.run(coro)

        return episodes

    def transform_trajectory_groups_to_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> list[tinker.Datum]:
        """Transform trajectory groups to Tinker Datum format.

        Note: For Tinker, advantage computation is done during this transformation step
        via transform_trajectory_groups_to_datums, which calls compute_advantage internally.

        Args:
            trainer_state: The trainer state containing trajectory_groups.
            **kwargs: Additional arguments.

        Returns:
            List of Tinker Datum objects.
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"

        # For Tinker, we don't compute advantages here - it's done in the combined transform
        # Just return an empty list as placeholder; actual datums are created in process_backend_batch
        return []

    def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
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
        assert self.policy_trainer.training_client is not None, "training_client is not initialized"
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"

        # Clear previous training data
        self._training_datums = []
        self._training_logprobs = []

        # Get the event loop from extra_info if available
        loop = trainer_state.extra_info.get("loop")
        if loop is None:
            raise RuntimeError("Event loop not found in trainer_state.extra_info")

        # Run the forward-backward pass asynchronously
        coro = self._process_backend_batch_async(trainer_state)
        asyncio.run_coroutine_threadsafe(coro, loop).result()

    async def _process_backend_batch_async(self, trainer_state: TrainerState) -> None:
        """Async implementation of process_backend_batch.

        Delegates to TinkerPolicyTrainer.forward_backward_from_trajectory_groups.
        """
        assert self.policy_trainer is not None
        trajectory_groups = trainer_state.trajectory_groups

        # Get algorithm config from trainer state or use default
        algorithm_config = trainer_state.extra_info.get("algorithm_config")

        # Use TinkerPolicyTrainer's method for forward-backward
        training_datums, training_logprobs = await self.policy_trainer.forward_backward_from_trajectory_groups(
            trajectory_groups,
            algorithm_config=algorithm_config,
        )

        # Store for metrics computation
        self._training_datums = training_datums
        self._training_logprobs = training_logprobs

        # Store datums as backend batch
        trainer_state.backend_batch = training_datums

    def compute_advantages(
        self,
        trainer_state: TrainerState,
        algorithm_config: AlgorithmConfig,
        **kwargs,
    ) -> None:
        """Compute advantages from trajectory groups.

        For Tinker, advantage computation is done in process_backend_batch via
        transform_trajectory_groups_to_datums. This method is a no-op but stores
        the algorithm config for use in process_backend_batch.
        """
        # Store algorithm config for use in process_backend_batch
        trainer_state.extra_info["algorithm_config"] = algorithm_config

    def update_policy(self, trainer_state: TrainerState) -> None:
        """Update the policy via optimizer step.

        For Tinker, this performs the optimizer step after forward-backward.

        Args:
            trainer_state: The trainer state.
        """
        assert self.policy_trainer is not None, "policy_trainer is not initialized"

        # Get the event loop from extra_info
        loop = trainer_state.extra_info.get("loop")
        if loop is None:
            raise RuntimeError("Event loop not found in trainer_state.extra_info")

        # Run the optimizer step asynchronously
        coro = self._update_policy_async(trainer_state)
        asyncio.run_coroutine_threadsafe(coro, loop).result()

        # Compute KL metrics if we have training data
        if self._training_datums and self._training_logprobs:
            kl_metrics = compute_kl_and_entropy_metrics(
                self._training_datums,
                self._training_logprobs,
            )
            trainer_state.metrics.update(kl_metrics)

    async def _update_policy_async(self, trainer_state: TrainerState) -> None:
        """Async implementation of update_policy."""
        assert self.policy_trainer is not None

        learning_rate = self.full_config.training.learning_rate
        beta1 = self.full_config.training.get("beta1", 0.9)
        beta2 = self.full_config.training.get("beta2", 0.95)
        eps = self.full_config.training.get("eps", 1e-8)

        # Optimizer step
        optim_step_future = await self.policy_trainer.optim_step_future(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        await optim_step_future.result_async()

    # =========================================================================
    # Hook methods
    # =========================================================================

    def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        Initializes the policy trainer and loads checkpoints if available.
        """
        # Ensure checkpoint directory exists
        os.makedirs(self.full_config.trainer.default_local_dir, exist_ok=True)

        # Get the event loop from extra_info
        loop = trainer_state.extra_info.get("loop")
        if loop is None:
            raise RuntimeError("Event loop not found in trainer_state.extra_info. The UnifiedTrainer should set this.")

        # Initialize policy trainer
        self.policy_trainer = TinkerPolicyTrainer(
            config=self.full_config,
            service_client=self.service_client,
        )

        # Initialize training client and load checkpoint
        coro = self.policy_trainer.initialize_async(resume_from_checkpoint=True)
        start_batch, self.sampling_client = asyncio.run_coroutine_threadsafe(coro, loop).result()

        # Update trainer state with the start batch from checkpoint
        self._start_batch = start_batch
        trainer_state.global_step = start_batch

    def on_train_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of training."""
        # Final checkpoint save
        loop = trainer_state.extra_info.get("loop")
        if loop is not None and self.policy_trainer is not None:
            coro = self.policy_trainer.save_checkpoint_async(trainer_state.global_step, kind="state")
            asyncio.run_coroutine_threadsafe(coro, loop).result()

    def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of each batch. Nothing to do for Tinker."""
        pass

    def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch.

        Saves checkpoint and updates sampling client.
        """
        loop = trainer_state.extra_info.get("loop")
        if loop is None or self.policy_trainer is None:
            return

        global_step = trainer_state.global_step

        # Save sampler checkpoint after each batch
        coro = self.policy_trainer.save_checkpoint_async(global_step, kind="sampler")
        path_dict = asyncio.run_coroutine_threadsafe(coro, loop).result()
        self.sampling_client = self.policy_trainer.create_sampling_client(path_dict["sampler_path"])

        # Save full state checkpoint periodically
        save_freq = self.full_config.trainer.get("save_freq", 0)
        if save_freq > 0 and global_step % save_freq == 0:
            logger.info(f"Saving state checkpoint at step {global_step}")
            coro = self.policy_trainer.save_checkpoint_async(global_step, kind="state")
            asyncio.run_coroutine_threadsafe(coro, loop).result()

        # Print metrics table
        if trainer_state.metrics:
            print_metrics_table(trainer_state.metrics, global_step)

    def on_epoch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of an epoch."""
        logger.info(f"Starting epoch {trainer_state.epoch}")

    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of an epoch."""
        logger.info(f"Completed epoch {trainer_state.epoch}")

    def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation.

        Returns:
            bool: True if validation should proceed, False otherwise.
        """
        trainer_state.is_training = False
        return True

    def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
