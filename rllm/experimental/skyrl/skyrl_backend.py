"""
SkyRL backend implementation for the UnifiedTrainer.

This backend implements the BackendProtocol interface to provide
SkyRL-specific implementations for the unified training pipeline.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.experimental.common.advantage import AlgorithmConfig
from rllm.experimental.protocol import BackendProtocol

if TYPE_CHECKING:
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.trainer import RayPPOTrainer
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.generators.base import GeneratorInterface

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


class SkyRLBackend(BackendProtocol[Iterable, TrainingInputBatch]):
    """
    SkyRL backend for the unified trainer.

    This backend provides:
        - SkyRL-compatible rollout engine
        - Policy training via SkyRL's RayPPOTrainer
        - Integration with SkyRL's GeneratorInterface

    The backend uses async methods naturally to match SkyRL's async API.
    """

    name: str = "skyrl"
    requires_loop: bool = True  # SkyRL uses async operations

    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        """Initialize the SkyRLBackend.

        Args:
            config: The full configuration object.
            **kwargs: Additional arguments.
        """
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference
        self.full_config = config

        # SkyRL components - will be initialized in init_rollout_engine and on_train_start
        self.skyrl_trainer: RayPPOTrainer | None = None
        self.inference_engine_client: InferenceEngineClient | None = None
        self.generator: GeneratorInterface | None = None
        self.tokenizer: AutoTokenizer | None = None

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: RolloutEngine | None = None

        # Store algorithm config for use in process_backend_batch
        self._algorithm_config: AlgorithmConfig | None = None

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the SkyRL-compatible rollout engine.

        For SkyRL, we need to set up the inference engine client and generator.
        The actual rollout engine will be a wrapper that uses SkyRL's components.

        Args:
            **kwargs: Additional arguments, including the various configurations

        Returns:
            RolloutEngine: The initialized rollout engine.
        """
        # Import here to avoid hard dependency
        from rllm.engine.rollout.skyrl_engine import SkyRLEngine

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.full_config.model.name)

        # The rollout engine will be fully initialized in on_train_start
        # when we have access to the SkyRL trainer components
        self.rollout_engine = SkyRLEngine(
            config=self.full_config,
            tokenizer=self.tokenizer,
        )
        return self.rollout_engine

    def validate_config(self) -> None:
        """Validate SkyRL-specific configuration settings."""
        # Add any SkyRL-specific validation here
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

    def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        super().shutdown()
        # Cleanup SkyRL components if needed
        if self.skyrl_trainer is not None:
            # SkyRL trainer cleanup if needed
            pass

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

        For SkyRL backend, this function handles:
        1. Building an interleaved batch (each task repeated `group_size` times)
        2. Executing tasks using the agent workflow engine

        Args:
            batch: Input batch (list of task dicts from dataloader).
            agent_workflow_engine: The workflow engine to use for episode generation.
            **kwargs: Additional arguments.

        Returns:
            List of generated episodes.
        """
        assert self.rollout_engine is not None, "rollout_engine is not initialized"

        # Build interleaved batch
        group_size = self.full_config.training.get("group_size", 1)
        interleaved_batch = _build_interleave_batch(batch, group_size)

        # Extract task IDs
        task_ids = [item["uid"] for item in interleaved_batch]

        # Execute tasks using the agent workflow engine (async)
        episodes = await agent_workflow_engine.execute_tasks(interleaved_batch, task_ids, **kwargs)

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
        Note: This is async for protocol compatibility but operations are sync (blocking)

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        assert self.skyrl_trainer is not None, "skyrl_trainer is not initialized"
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Use SkyRL's forward/backward pass
        # This computes log probs, values, and processes rewards
        processed = self.skyrl_trainer.fwd_logprobs_values_reward(training_input)

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
        assert self.skyrl_trainer is not None, "skyrl_trainer is not initialized"
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Use SkyRL's advantage computation
        # This updates the training_input with advantages and returns
        training_input = self.skyrl_trainer.compute_advantages_and_returns(training_input)

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

        Args:
            trainer_state: The trainer state.
            **kwargs: Additional arguments.
        """
        assert self.skyrl_trainer is not None, "skyrl_trainer is not initialized"
        assert trainer_state.backend_batch is not None, "Backend batch is not set"

        training_input: TrainingInputBatch = trainer_state.backend_batch

        # Use SkyRL's training step
        # This updates both critic and policy
        self.skyrl_trainer.train_critic_and_policy(training_input)

    # =========================================================================
    # Async hook methods
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training.

        Initializes the SkyRL trainer and related components.
        """
        # Import here to avoid hard dependency
        from skyrl_train.trainer import RayPPOTrainer
        from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
        from skyrl_train.utils.tracking import Tracking

        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.full_config.model.name)

        # Initialize SkyRL components
        # Note: This is a simplified initialization - you may need to adjust
        # based on your specific SkyRL setup and configuration structure
        # The actual initialization should be done based on your SkyRL config structure

        # For now, we'll defer full initialization until the components are actually needed
        # This allows for more flexible setup based on your specific SkyRL configuration
        # You should implement the actual initialization based on:
        # - config.inference_engine settings for InferenceEngineClient
        # - config.generator settings for GeneratorInterface
        # - Your specific SkyRL setup

        # Placeholder - you'll need to implement based on your SkyRL config
        logger.warning(
            "SkyRLBackend.on_train_start() needs to be fully implemented based on "
            "your SkyRL configuration structure. You'll need to initialize: "
            "1. InferenceEngineClient from config.inference_engine "
            "2. GeneratorInterface (or use UnifiedWorkflowEngine) "
            "3. RayPPOTrainer with appropriate arguments"
        )

        # Update rollout engine with SkyRL components
        if self.rollout_engine is not None:
            self.rollout_engine.set_skyrl_components(
                inference_engine_client=self.inference_engine_client,
                trainer=self.skyrl_trainer,
            )

        # Load checkpoint if resuming
        if self.full_config.trainer.get("resume_mode", "none") != "none":
            self.skyrl_trainer.global_step = self.skyrl_trainer.load_checkpoints()

        trainer_state.global_step = self.skyrl_trainer.global_step

    async def on_train_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of training."""
        if self.skyrl_trainer is not None:
            # Save final checkpoint if needed
            save_freq = self.full_config.rllm.trainer.get("save_freq", 0)
            if save_freq > 0 and trainer_state.global_step % save_freq != 0:
                logger.info(f"Saving final checkpoint at step {trainer_state.global_step}")
                self.skyrl_trainer.save_checkpoints()

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch.

        Saves checkpoint and syncs weights if needed.
        """
        if self.skyrl_trainer is None:
            return

        global_step = trainer_state.global_step

        # Save checkpoint periodically
        save_freq = self.full_config.rllm.trainer.get("save_freq", 0)
        if save_freq > 0 and global_step % save_freq == 0:
            logger.info(f"Saving checkpoint at step {global_step}")
            self.skyrl_trainer.save_checkpoints()

        # Sync weights to inference engines if needed
        if self.full_config.trainer.get("colocate_all", False):
            # Sync policy weights to inference engines
            import ray
            ray.get(self.skyrl_trainer.sync_policy_weights_to_inference_engines())

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

