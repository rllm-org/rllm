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
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.generators.base import GeneratorInterface

    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


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
        skyrl_trainer=None,
        inference_engine_client=None,
        tokenizer=None,
        **kwargs,
    ):
        """Initialize the SkyRLBackend.

        Args:
            config: The full configuration object.
            skyrl_trainer: Initialized SkyrlTrainer instance (required).
            inference_engine_client: Initialized InferenceEngineClient instance (required).
            tokenizer: Tokenizer instance (required).
            **kwargs: Additional arguments.
        """
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference
        self.full_config = config

        # SkyRL components - must be provided by launcher
        self.skyrl_trainer = skyrl_trainer
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer

        # Validate required components
        if self.skyrl_trainer is None:
            raise ValueError("skyrl_trainer must be provided. Initialize it in the launcher.")
        if self.inference_engine_client is None:
            raise ValueError("inference_engine_client must be provided. Initialize it in the launcher.")
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided. Initialize it in the launcher.")

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: RolloutEngine | None = None

        # Store workflow engine (passed from UnifiedTrainer)
        self.workflow_engine: UnifiedWorkflowEngine | None = None

        # Store algorithm config for use in process_backend_batch
        self._algorithm_config: AlgorithmConfig | None = None

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
        from rllm.engine.rollout.skyrl_engine import SkyRLEngine

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
        # Teardown inference engine client (async operation in sync method)
        # UnifiedTrainer.shutdown() stops the event loop, so we create a new one for teardown
        if self.inference_engine_client is not None:
            import asyncio
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
        """Generate episodes using SkyrlTrainer's generator.

        For SkyRL backend, this function:
        1. Builds interleaved batch (each task repeated `group_size` times)
        2. Uses SkyrlTrainer.generate() which internally uses RLLMGenerator
        3. Converts GeneratorOutput to Episodes (for compatibility with unified trainer)

        Args:
            batch: Input batch (list of task dicts from dataloader).
            agent_workflow_engine: UnifiedWorkflowEngine for running workflows.
            **kwargs: Additional arguments including trainer_state for global_step.

        Returns:
            List of generated episodes.
        """
        from skyrl_train.generators.base import GeneratorInput, GeneratorOutput, BatchMetadata, TrajectoryID
        from skyrl_train.generators.utils import prepare_generator_input
        from rllm.agents.agent import Episode, Trajectory, Step
        from rllm.engine.rollout import ModelOutput

        # Store workflow engine for reference
        self.workflow_engine = agent_workflow_engine
        
        # Update generator's workflow engine if it's an RLLMGenerator
        # (The generator was created in the launcher, but the workflow engine is only available now)
        if hasattr(self.skyrl_trainer, 'generator') and hasattr(self.skyrl_trainer.generator, 'workflow_engine'):
            self.skyrl_trainer.generator.workflow_engine = agent_workflow_engine

        # Get global step from kwargs (passed by unified trainer)
        global_step = kwargs.get("global_step", 0)
        is_training = kwargs.get("is_training", True)
        training_phase = "train" if is_training else "eval"

        # Get sampling params from config
        sampling_params = self.full_config.get("sampling", {})
        default_env_class = self.full_config.get("environment", {}).get("env_class", "BaseTextEnv")
        group_size = self.full_config.training.get("group_size", 1)

        # Build interleaved batch (each task repeated `group_size` times) - inline like Verl
        batch_with_uid = []
        for batch_item in batch:
            batch_with_uid.append({**batch_item, "uid": str(uuid.uuid4())})
        
        interleaved_batch = []
        for batch_item in batch_with_uid:
            interleaved_batch.extend([batch_item for _ in range(group_size)])

        # Prepare GeneratorInput from interleaved batch
        generator_input, uids = prepare_generator_input(
            prompts=interleaved_batch,
            n_samples_per_prompt=1,  # Already interleaved, so 1 sample per prompt
            sampling_params=sampling_params,
            default_env_class=default_env_class,
            training_phase=training_phase,
            global_step=global_step,
        )

        # Call SkyrlTrainer.generate() which uses RLLMGenerator internally
        generator_output: GeneratorOutput = await self.skyrl_trainer.generate(generator_input)

        # Convert GeneratorOutput to Episodes (for unified trainer compatibility)
        episodes = []
        prompt_token_ids = generator_output["prompt_token_ids"]
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]
        loss_masks = generator_output["loss_masks"]
        rollout_logprobs = generator_output.get("rollout_logprobs")

        for i, (prompt_tokens, response_tokens, reward, loss_mask) in enumerate(
            zip(prompt_token_ids, response_ids, rewards, loss_masks)
        ):
            # Get trajectory_id if available
            trajectory_id = generator_input.get("trajectory_ids", [None] * len(prompt_token_ids))[i]
            if trajectory_id:
                task_id = trajectory_id.instance_id
                rollout_idx = trajectory_id.repetition_id
                episode_id = f"{task_id}:{rollout_idx}"
            else:
                episode_id = uids[i] if i < len(uids) else f"task_{i}:0"

            # Create Step from GeneratorOutput
            step = Step(
                prompt_ids=prompt_tokens,
                response_ids=response_tokens,
                logprobs=rollout_logprobs[i] if rollout_logprobs and i < len(rollout_logprobs) else [],
                reward=reward if isinstance(reward, (int, float)) else reward[-1] if isinstance(reward, list) and len(reward) > 0 else 0.0,
                done=True,
            )

            # Create ModelOutput for the step
            step.model_output = ModelOutput(
                prompt_ids=prompt_tokens,
                completion_ids=response_tokens,
                logprobs=rollout_logprobs[i] if rollout_logprobs and i < len(rollout_logprobs) else None,
            )

            # Create Trajectory
            trajectory = Trajectory(
                uid=episode_id,
                name="default_traj_name",
                task=interleaved_batch[i] if i < len(interleaved_batch) else {},
                steps=[step],
                reward=reward if isinstance(reward, (int, float)) else reward[-1] if isinstance(reward, list) and len(reward) > 0 else 0.0,
            )

            # Create Episode
            episode = Episode(
                id=episode_id,
                task=interleaved_batch[i] if i < len(interleaved_batch) else {},
                trajectories=[trajectory],
                is_correct=reward > 0.0 if isinstance(reward, (int, float)) else (reward[-1] > 0.0 if isinstance(reward, list) and len(reward) > 0 else False),
            )

            episodes.append(episode)

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
        After training, we sync weights to inference engines (following SkyRL's pattern).

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

        Sets global_step from the SkyrlTrainer if available.
        """
        if self.skyrl_trainer is not None:
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

        Saves checkpoint and syncs weights to inference engines.
        This follows the pattern where batch-level operations (like checkpointing in Verl)
        happen in on_batch_end.
        """
        if self.skyrl_trainer is None:
            return

        global_step = trainer_state.global_step

        # Save checkpoint periodically
        save_freq = self.full_config.rllm.trainer.get("save_freq", 0)
        if save_freq > 0 and global_step % save_freq == 0:
            logger.info(f"Saving checkpoint at step {global_step}")
            self.skyrl_trainer.save_checkpoints()

        # Sync weights to inference engines after training (following SkyRL's pattern)
        # This ensures inference engines use the latest policy weights for the next batch's generation
        colocate_all = self.full_config.trainer.get("colocate_all", False)
        
        if colocate_all:
            # Offload policy model optimizer to CPU (keep model on GPU for now)
            self.skyrl_trainer.policy_model.offload_to_cpu(offload_optimizer=True, offload_model=False)
            # Wake up inference engines for weight loading
            await self.inference_engine_client.wake_up(tags=["weights"])
        
        # Sync weights from policy model to inference engines (always, not just for colocated)
        import ray
        ray.get(self.skyrl_trainer.sync_policy_weights_to_inference_engines())
        
        if colocate_all:
            # Offload policy model to CPU (free GPU for inference)
            self.skyrl_trainer.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
            # Wake up inference engines for KV cache loading
            await self.inference_engine_client.wake_up(tags=["kv_cache"])

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

