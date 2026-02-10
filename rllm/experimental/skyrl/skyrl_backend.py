"""
SkyRL backend implementation for the UnifiedTrainer.

This backend implements the BackendProtocol interface to provide
SkyRL-specific implementations for the unified training pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from skyrl_train.training_batch import TrainingInputBatch

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.experimental.common.advantage import AlgorithmConfig
from rllm.experimental.protocol import BackendProtocol

if TYPE_CHECKING:
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState

# Import RayPPOTrainer for inheritance
from skyrl_train.trainer import RayPPOTrainer

logger = logging.getLogger(__name__)


class SkyRLBackend(BackendProtocol[Iterable, TrainingInputBatch], RayPPOTrainer):
    """
    SkyRL backend for the unified trainer.

    Inherits from both BackendProtocol and RayPPOTrainer to:
        - Provide the BackendProtocol interface for UnifiedTrainer
        - Reuse RayPPOTrainer's training infrastructure and utilities (e.g. model building, checkpointing)

    The backend uses async methods naturally to match SkyRL's async API.
    """

    name: str = "skyrl"
    requires_loop: bool = True  # SkyRL uses async operations

    def __init__(
        self,
        config: DictConfig,
        tracker,
        tokenizer,
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=None,
        colocate_pg=None,
        **kwargs,
    ):
        """Initialize the SkyRLBackend.

        Args:
            config: The full configuration object.
            tracker: Tracking instance for experiment tracking.
            tokenizer: Tokenizer instance.
            train_dataset: Training PromptDataset (optional).
            eval_dataset: Evaluation PromptDataset (optional).
            inference_engine_client: Initialized InferenceEngineClient instance.
            generator: Unused. Passed through to RayPPOTrainer.__init__() which requires it.
            colocate_pg: Optional placement group for colocated training.
            **kwargs: Additional arguments.
        """
        # Initialize RayPPOTrainer first - this sets up all trainer infrastructure
        RayPPOTrainer.__init__(
            self,
            cfg=config,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

        # Initialize BackendProtocol
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference (RayPPOTrainer uses self.cfg)
        self.full_config = config

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: RolloutEngine | None = None

        # Store workflow engine (passed from UnifiedTrainer)
        self.workflow_engine: UnifiedWorkflowEngine | None = None

        # Store algorithm config for use in process_backend_batch
        self._algorithm_config: AlgorithmConfig | None = None

        # Track whether SkyRL weight-sync sender/receiver state is initialized.
        # Native SkyRL trainer initializes this once before training starts.
        self._weight_sync_state_initialized = False

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the SkyRL-compatible rollout engine.

        Args:
            **kwargs: Additional arguments, including the various configurations

        Returns:
            RolloutEngine: The initialized rollout engine.
        """
        # Import here to avoid hard dependency
        from rllm.experimental.rollout.skyrl_engine import SkyRLEngine

        # Create rollout engine with provided components
        # Note: inference_engine_client is already set in __init__, so no need for set_skyrl_components
        self.rollout_engine = SkyRLEngine(
            inference_engine_client=self.inference_engine_client,
            tokenizer=self.tokenizer,
            config=self.full_config,
        )

        return self.rollout_engine

    def validate_config(self) -> None:
        """Validate SkyRL-specific configuration settings."""
        # Validation done during initialization. #TODO: add assert for async or rewards?
        pass

    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> Iterable:
        """Get dataloader for the given dataset.

        For SkyRL, we create standard PyTorch DataLoaders.

        Args:
            dataset: The dataset to create dataloader from.
            trainer_state: The trainer state.

        Returns:
            DataLoader wrapped dataset.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None for SkyRLBackend")

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

    def transform_to_backend_batch(self, trainer_state, **kwargs) -> TrainingInputBatch:
        """Transform rllm-native data structures to SkyRL TrainingInputBatch format.

        This method expects `trainer_state.trajectory_groups` to be set by the
        previous pipeline stage and delegates the actual conversion to
        `transform_trajectory_groups_to_backend_batch` which handles the
        construction of a SkyRL `TrainingInputBatch`.
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        assert self.rollout_engine is not None, "rollout_engine is not initialized."

        return self.transform_trajectory_groups_to_backend_batch(trainer_state, **kwargs)

    def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        # Teardown inference engine client (async operation in sync method)
        # UnifiedTrainer.shutdown() stops the event loop, so we create a new one for teardown
        if self.inference_engine_client is not None:
            asyncio.run(self.inference_engine_client.teardown())

    # =========================================================================
    # Async pipeline methods
    # =========================================================================

    async def generate_episodes(
        self,
        batch: Any,
        agent_workflow_engine: UnifiedWorkflowEngine,
        **kwargs,
    ) -> list[Episode]:
        """Generate episodes by calling the workflow engine directly.

        The workflow engine returns full Episodes with all metadata preserved
        (termination_reason, metrics, trajectory names, step-level fields).

        Uses `rllm.rollout.n` for training and `rllm.rollout.n_val` for validation
        (consistent with VERL backend pattern).

        Args:
            batch: Input batch (list of task dicts from dataloader).
            agent_workflow_engine: UnifiedWorkflowEngine for running workflows.
            **kwargs: Additional arguments including global_step.

        Returns:
            List of generated episodes with full metadata.
        """
        # Store workflow engine for reference
        self.workflow_engine = agent_workflow_engine

        # Get global step from kwargs (passed by unified trainer)
        global_step = kwargs.get("global_step", 0)

        # Determine training phase from workflow engine mode
        training_phase = agent_workflow_engine.current_mode if hasattr(agent_workflow_engine, "current_mode") else "train"
        is_validation = training_phase != "train"

        # Use rllm.rollout.n for training, rllm.rollout.n_val for validation (VERL pattern)
        if training_phase == "train":
            group_size = self.full_config.rllm.rollout.n
        else:
            group_size = self.full_config.rllm.rollout.n_val

        # Set training step for episode logging
        steps_per_epoch = len(self.train_dataloader) if self.train_dataloader else 1
        epoch = (global_step - 1) // steps_per_epoch if global_step > 0 else 0
        agent_workflow_engine.set_training_step(global_step, mode=training_phase, epoch=epoch)

        # Build interleaved tasks directly from batch (no SkyRL format conversion needed)
        tasks = []
        task_ids = []
        for item in batch:
            uid = item.get("uid") or item.get("unique_id") or str(uuid.uuid4())
            for _ in range(group_size):
                tasks.append(item)
                task_ids.append(uid)

        # Execute workflows directly - returns full Episodes with all metadata
        episodes: list[Episode] = await agent_workflow_engine.execute_tasks(
            tasks, task_ids, is_validation=is_validation,
        )

        # Compute rollout metrics from episodes
        if not hasattr(self, "all_metrics"):
            self.all_metrics = {}

        all_rewards = []
        for ep in episodes:
            for traj in ep.trajectories:
                if traj.reward is not None:
                    all_rewards.append(traj.reward)

        if all_rewards:
            self.all_metrics["rollout/mean_raw_reward"] = sum(all_rewards) / len(all_rewards)
            self.all_metrics["rollout/num_episodes"] = len(episodes)
            self.all_metrics["rollout/num_trajectories"] = len(all_rewards)

        return episodes

    def transform_trajectory_groups_to_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> TrainingInputBatch:
        """Transform trajectory groups to SkyRL TrainingInputBatch format.

        Args:
            trainer_state: The trainer state containing trajectory_groups.
            **kwargs: Additional arguments.

        Returns:
            TrainingInputBatch: SkyRL's training input format.
        """
        from rllm.experimental.skyrl.transform import transform_trajectory_groups_to_training_input

        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        assert self.rollout_engine is not None, "rollout_engine is not initialized."

        trajectory_groups: list[TrajectoryGroup] = trainer_state.trajectory_groups
        return transform_trajectory_groups_to_training_input(
            trajectory_groups,
            self.rollout_engine,
            self.full_config.data.max_prompt_length,
            self.full_config.data.max_response_length,
        )

    async def process_backend_batch(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Process the backend batch by running forward-backward pass.

        For SkyRL, this performs:
        1. Forward pass to compute log probs and values
        2. Compute rewards if needed
        3. Store results back in trainer_state.backend_batch

        Reuses logic from SkyRL's RayPPOTrainer.

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Use SkyRL's forward/backward pass
        # This computes log probs, values, and processes rewards
        # Wrap in asyncio.to_thread() to avoid blocking the event loop
        processed = await asyncio.to_thread(self.fwd_logprobs_values_reward, training_input)

        trainer_state.backend_batch = processed

    async def compute_advantages(
        self,
        trainer_state: TrainerState,
        algorithm_config: AlgorithmConfig,
        **kwargs,
    ) -> None:
        """Compute advantages from trajectory groups.

        For SkyRL, advantage computation is done using SkyRL's trainer.
        Note: This is async for protocol compatibility but operations are sync.

        Args:
            trainer_state: The trainer state.
            algorithm_config: Algorithm configuration.
            **kwargs: Additional arguments.
        """
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Use SkyRL's advantage computation
        # This updates the training_input with advantages and returns
        # Wrap in asyncio.to_thread() to avoid blocking the event loop
        training_input = await asyncio.to_thread(self.compute_advantages_and_returns, training_input)

        # Remove rewards key as it's no longer needed after advantage computation
        if "rewards" in training_input:
            training_input.pop("rewards")

        trainer_state.backend_batch = training_input

    async def update_policy(
        self,
        trainer_state: TrainerState,
        **kwargs,
    ) -> None:
        """Update the policy via optimizer step.

        For SkyRL, this performs the optimizer step after forward-backward.
        After training, we sync weights to inference engines (following SkyRL's pattern).

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Keep RayPPOTrainer's internal global_step in sync with UnifiedTrainer.
        # SkyRL workers read this from training_input.metadata["global_step"] and
        # checkpoint naming also relies on self.global_step.
        self.global_step = trainer_state.global_step

        # Use SkyRL's training step
        # This updates both critic and policy
        # Wrap in asyncio.to_thread() to avoid blocking the event loop (contains ray.get() calls)
        await asyncio.to_thread(self.train_critic_and_policy, training_input)

        # Ensure all_metrics exists (RayPPOTrainer should initialize it, but be safe)
        if not hasattr(self, "all_metrics"):
            self.all_metrics = {}

        # Extract metrics from SkyRL's all_metrics (populated by train_critic_and_policy)
        # SkyRL's train_critic_and_policy updates self.all_metrics with:
        # - critic/* metrics from critic_statuses[0].metadata["train_status"]
        # - policy/* metrics from policy_statuses[0].metadata["train_status"]
        # Convert SkyRL metrics to rLLM logger format (training/ prefix for training metrics)
        if self.all_metrics:
            for key, value in self.all_metrics.items():
                # Skip non-scalar values and None
                if value is None:
                    continue

                # Convert value to scalar
                if isinstance(value, int | float):
                    scalar_value = value
                elif hasattr(value, "item"):  # torch.Tensor
                    try:
                        scalar_value = value.item()
                    except (ValueError, RuntimeError):
                        continue
                else:
                    continue

                # Map SkyRL metric keys to rLLM logger format
                # SkyRL uses: policy/*, critic/*, loss/*, reward/*
                # rLLM format: training/policy/*, training/critic/*, training/loss/*, reward/* (unchanged)
                if key.startswith("policy/") or key.startswith("critic/") or key.startswith("loss/"):
                    # Add training/ prefix for training-related metrics
                    rllm_key = f"training/{key}"
                elif key.startswith("reward/"):
                    # Keep reward metrics as-is (already in rLLM format)
                    rllm_key = key
                elif key.startswith("trainer/"):
                    # Map trainer/* to training/*
                    rllm_key = key.replace("trainer/", "training/", 1)
                else:
                    # For other metrics, add training/ prefix by default
                    rllm_key = f"training/{key}"

                trainer_state.metrics[rllm_key] = scalar_value

            # Clear all_metrics after extracting (SkyRL will repopulate on next batch)
            self.all_metrics.clear()

        # Add reward metrics from episodes/trajectory_groups if available
        # This ensures we have reward metrics even if all_metrics is empty
        if hasattr(trainer_state, "trajectory_groups") and trainer_state.trajectory_groups:
            import numpy as np

            from rllm.experimental.common.metrics import reduce_metrics_by_trajectory_name

            # Add reward metrics from trajectory groups
            reward_metrics = reduce_metrics_by_trajectory_name(trainer_state.trajectory_groups, prefix="reward")
            trainer_state.metrics.update(reward_metrics)

            # Also compute basic reward stats
            all_rewards = []
            for group in trainer_state.trajectory_groups:
                for traj in group.trajectories:
                    if traj.reward is not None:
                        all_rewards.append(traj.reward)

            if all_rewards:
                trainer_state.metrics["reward/mean"] = np.mean(all_rewards)
                trainer_state.metrics["reward/max"] = np.max(all_rewards)
                trainer_state.metrics["reward/min"] = np.min(all_rewards)
                trainer_state.metrics["reward/std"] = np.std(all_rewards)

        # Add basic training metrics that should always be present (rLLM format)
        # Get learning rate from optimizer if available
        if hasattr(self, "policy_model") and hasattr(self.policy_model, "optimizer"):
            optimizer = self.policy_model.optimizer
            if optimizer is not None and len(optimizer.param_groups) > 0:
                trainer_state.metrics["optim/lr"] = optimizer.param_groups[0].get("lr", 0.0)

        # Add global step and epoch (always present, rLLM format)
        trainer_state.metrics["training/global_step"] = trainer_state.global_step
        trainer_state.metrics["training/epoch"] = trainer_state.epoch

        # Debug logging to verify metrics are being populated
        num_training_metrics = len([k for k in trainer_state.metrics.keys() if k.startswith("training/") or k.startswith("reward/") or k.startswith("optim/")])
        logger.info(f"Step {trainer_state.global_step}: Extracted {num_training_metrics} training metrics. Keys: {sorted(trainer_state.metrics.keys())}")

        # Log warning if no metrics were found (for debugging)
        if len(trainer_state.metrics) <= 3:  # Only global_step, epoch, and maybe lr
            logger.warning(f"Step {trainer_state.global_step}: No training metrics found in all_metrics. all_metrics keys were: {list(self.all_metrics.keys()) if hasattr(self, 'all_metrics') and self.all_metrics else 'empty'}. Available trainer_state.metrics: {list(trainer_state.metrics.keys())}")

    async def _sync_policy_weights_for_next_rollout(self) -> None:
        """Sync policy weights to inference engines for subsequent generation."""
        import ray

        # In colocated mode, we need staged wake-up/offload for memory safety.
        if self.colocate_all:
            self.policy_model.offload_to_cpu(offload_optimizer=True, offload_model=False)
            await self.inference_engine_client.wake_up(tags=["weights"])

        try:
            weight_sync_refs = self.sync_policy_weights_to_inference_engines()
            await asyncio.to_thread(ray.get, weight_sync_refs)
        except AttributeError as e:
            if "_weight_transfer_sender" in str(e):
                logger.warning("Weight transfer sender not initialized. Skipping weight sync. Inference engines may use stale weights.")
            else:
                raise

        if self.colocate_all:
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await self.inference_engine_client.wake_up(tags=["kv_cache"])

    # =========================================================================
    # Async hook methods
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        Sets global_step from the trainer and initializes weight-sync state.
        """
        trainer_state.global_step = self.global_step

        # Match native SkyRL trainer behavior:
        # initialize weight transfer sender/receivers once before training.
        if not self._weight_sync_state_initialized:
            logger.info("Initializing policy->inference weight sync state")
            await asyncio.to_thread(self.init_weight_sync_state)
            self._weight_sync_state_initialized = True

        # Ensure generation starts from the current policy weights.
        await self._sync_policy_weights_for_next_rollout()

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of training."""
        self.global_step = trainer_state.global_step

        # Save final checkpoint if needed
        save_freq = self.full_config.rllm.trainer.get("save_freq", 0)
        if save_freq > 0 and trainer_state.global_step % save_freq != 0:
            logger.info(f"Saving final checkpoint at step {trainer_state.global_step}")
            await asyncio.to_thread(self.save_checkpoints)

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch.

        Saves checkpoint and syncs weights to inference engines.
        This follows the pattern where batch-level operations (like checkpointing in Verl)
        happen in on_batch_end.
        """
        global_step = trainer_state.global_step
        self.global_step = global_step

        # Save checkpoint periodically
        save_freq = self.full_config.rllm.trainer.get("save_freq", 0)
        if save_freq > 0 and global_step % save_freq == 0:
            logger.info(f"Saving checkpoint at step {global_step}")
            await asyncio.to_thread(self.save_checkpoints)

        # Sync weights after each optimizer step so the next rollout uses fresh policy.
        # This must run for both colocated and non-colocated configurations.
        await self._sync_policy_weights_for_next_rollout()

    async def on_epoch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of an epoch."""
        logger.info(f"Starting epoch {trainer_state.epoch}")

    async def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of an epoch."""
        logger.info(f"Completed epoch {trainer_state.epoch}")

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation.

        Sets validation mode on both trainer_state and rollout_engine.
        Following VERL backend pattern for consistency.

        Returns:
            bool: True if validation should proceed, False otherwise.
        """
        trainer_state.is_training = False
        if self.rollout_engine is not None:
            self.rollout_engine.validate = True
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation.

        Resets validation mode on both trainer_state and rollout_engine.
        """
        trainer_state.is_training = True
        if self.rollout_engine is not None:
            self.rollout_engine.validate = False
