"""Policy training module for Tinker-based RL.

This module handles gradient updates, model checkpointing, and data processing.
It does NOT contain any environment or agent logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tinker
from omegaconf import OmegaConf
from tinker import types
from tinker_cookbook import checkpoint_utils

from rllm.trainer.tinker.tinker_data_processor import (
    TinkerAdvantageComputer,
    TinkerTrajectoryFilter,
    process_episodes,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class TinkerPolicyTrainer:
    """
    Handles policy updates via gradient descent.

    This class handles:
    - Training client management
    - Data processing (filtering, advantages, datum conversion)
    - Forward-backward passes
    - Optimizer steps
    - Checkpoint saving/loading

    It does NOT handle:
    - Environment or agent interactions
    - Trajectory collection
    - Sampling
    """

    def __init__(
        self,
        config,
        service_client: tinker.ServiceClient,
    ):
        """
        Initialize the policy trainer.

        Args:
            config: Training configuration (OmegaConf)
            service_client: Tinker service client
        """
        self.config = config
        self.service_client = service_client
        self.training_client = None

        # Initialize data processors
        self.advantage_computer = TinkerAdvantageComputer(config.algorithm)
        self.trajectory_filter = TinkerTrajectoryFilter(config.algorithm)

    async def initialize_async(self, resume_from_checkpoint: bool = True):
        """
        Initialize or resume training client.

        Args:
            resume_from_checkpoint: If True, attempt to resume from last checkpoint
        """
        # Check for existing checkpoint
        resume_info = None
        if resume_from_checkpoint:
            # Check if a Tinker model ID is provided in config
            tinker_model_id = OmegaConf.select(self.config, "trainer.resume_from_tinker_id", default=None)
            if tinker_model_id:
                # Parse Tinker model ID format: tinker://{uuid}/weights/{checkpoint_name}
                # Example: tinker://7af7f6f0-e8f9-4124-ab93-e7f08eb54a9d/weights/000060
                if not tinker_model_id.startswith("tinker://"):
                    raise ValueError(f"Invalid Tinker model ID format: {tinker_model_id}. Expected format: tinker://uuid/weights/checkpoint_name")

                # Extract checkpoint name (e.g., "000060") from path
                if "/weights/" not in tinker_model_id:
                    raise ValueError(f"Invalid Tinker model ID format: {tinker_model_id}. Expected format: tinker://uuid/weights/checkpoint_name")

                checkpoint_name = tinker_model_id.split("/weights/")[-1]
                try:
                    # Extract batch number from checkpoint name (6-digit format: 000060 -> 60)
                    batch = int(checkpoint_name)
                except ValueError as err:
                    raise ValueError(f"Invalid checkpoint name format: {checkpoint_name}. Expected 6-digit number (e.g., 000060)") from err

                # Construct sampler path by replacing "weights" with "sampler_weights"
                sampler_path = tinker_model_id.replace("/weights/", "/sampler_weights/")

                resume_info = {
                    "state_path": tinker_model_id,
                    "sampler_path": sampler_path,
                    "batch": batch,
                }
                logger.info(f"Resuming from Tinker model ID: {tinker_model_id}")
            else:
                # Fall back to local checkpoint lookup
                resume_info = self.get_last_checkpoint()

        if resume_info:
            # Resume from checkpoint
            logger.info(f"Resuming from checkpoint: {resume_info}")
            try:
                self.training_client = await self.service_client.create_training_client_from_state_async(resume_info["state_path"])
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                raise

            if "sampler_path" in resume_info:
                logger.info(f"Using sampler checkpoint: {resume_info['sampler_path']}")
                sampling_client = self.create_sampling_client(resume_info["sampler_path"])
            else:
                # Fallback: convert state path to sampler path
                sampler_path = resume_info["state_path"].replace("weights", "sampler_weights")
                logger.info(f"No sampler_path in checkpoint, using: {sampler_path}")
                sampling_client = self.create_sampling_client(sampler_path)

            # Checkpoint batch number represents the last completed batch
            # So we should resume from the next batch (batch + 1)
            checkpoint_batch = resume_info["batch"]
            start_batch = checkpoint_batch + 1
            logger.info(f"Resuming from checkpoint at batch {checkpoint_batch}, starting training at batch {start_batch}")
            return start_batch, sampling_client
        else:
            # Start from scratch
            # Create LoRA training client
            # Configure which layers to train (for compatibility with deployment targets)
            train_unembed = OmegaConf.select(self.config, "model.train_unembed", default=True)
            train_attn = OmegaConf.select(self.config, "model.train_attn", default=True)
            train_mlp = OmegaConf.select(self.config, "model.train_mlp", default=True)

            self.training_client = await self.service_client.create_lora_training_client_async(
                base_model=self.config.model.name,
                rank=self.config.model.lora_rank,
                train_unembed=train_unembed,
                train_attn=train_attn,
                train_mlp=train_mlp,
            )
            logger.info(f"Starting training from scratch with model: {self.config.model.name}")
            sampler_future = await self.training_client.save_weights_for_sampler_async(name="000000")
            sampler_result = await sampler_future.result_async()
            sampling_client = self.create_sampling_client(sampler_result.path)
            return 0, sampling_client

    def _remove_mask(self, datum: tinker.Datum) -> tinker.Datum:
        """Remove mask from datum (not needed by forward_backward)."""
        return tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
        )

    async def step(
        self,
        episodes: list,
        learning_rate: float = None,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        optimizer_step: bool = True,
    ) -> tuple[list[torch.Tensor], list[tinker.Datum], dict]:
        """
        Complete training step: process episodes and update policy.

        This method:
        1. Processes episodes to compute advantages and create datums
        2. Performs forward-backward pass
        3. Applies optimizer step

        Args:
            episodes: List of Episode objects
            learning_rate: Learning rate (uses config value if None)
            optimizer_step: Whether to apply optimizer step after forward-backward

        Returns:
            Tuple of (training_logprobs, training_datums, grouping_metrics)
            - training_logprobs: List of training logprobs for KL computation
            - training_datums: List of datums WITH masks for metrics
            - grouping_metrics: Dict with grouping and advantage statistics
        """
        if learning_rate is None:
            learning_rate = self.config.training.learning_rate

        # Step 1: Process to datums (includes filtering and advantage computation)
        training_datums, grouping_metrics = process_episodes(
            episodes,
            self.advantage_computer,
            self.trajectory_filter,
            self.config.algorithm,
        )

        # Step 3: Remove mask from datums (not needed by forward_backward)
        datums_no_mask = [self._remove_mask(datum) for datum in training_datums]

        # Step 4: Forward-backward pass
        fwd_bwd_future = await self.training_client.forward_backward_async(
            datums_no_mask,
            loss_fn="importance_sampling",
        )

        # Step 5: Optimizer step
        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        if optimizer_step:
            optim_step_future = await self.training_client.optim_step_async(adam_params)

        # Wait for completion and extract logprobs
        fwd_bwd_result = await fwd_bwd_future.result_async()

        if optimizer_step:
            await optim_step_future.result_async()

        # Extract training logprobs from loss_fn_outputs
        training_logprobs_D = []
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs = output["logprobs"].to_torch()
            training_logprobs_D.append(training_logprobs)

        # Return logprobs, datums (with masks for metrics), and grouping metrics
        return training_logprobs_D, training_datums, grouping_metrics

    async def forward_backward_future(self, episodes: list):
        training_datums, grouping_metrics = process_episodes(
            episodes,
            self.advantage_computer,
            self.trajectory_filter,
            self.config.algorithm,
        )

        datums_no_mask = [self._remove_mask(datum) for datum in training_datums]

        fwd_bwd_future = await self.training_client.forward_backward_async(
            datums_no_mask,
            loss_fn="importance_sampling",
        )

        return fwd_bwd_future, grouping_metrics

    async def optim_step_future(self, learning_rate: float = None, beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8):
        if learning_rate is None:
            learning_rate = self.config.training.learning_rate

        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        optim_step_future = await self.training_client.optim_step_async(adam_params)
        return optim_step_future

    async def save_checkpoint_async(
        self,
        batch_idx: int,
        kind: str = "sampler",
    ) -> dict:
        """
        Save checkpoint and return paths.

        Args:
            batch_idx: Current batch index
            kind: Checkpoint kind ("state", "sampler", or "both")

        Returns:
            Dictionary with checkpoint paths
        """
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=f"{batch_idx:06d}",
            log_path=self.config.trainer.default_local_dir,
            kind=kind,
            loop_state={"batch": batch_idx},
        )
        return path_dict

    def create_sampling_client(self, sampler_path: str) -> tinker.SamplingClient:
        """
        Create a sampling client from a checkpoint path.

        Args:
            sampler_path: Path to sampler checkpoint

        Returns:
            Tinker sampling client
        """
        return self.training_client.create_sampling_client(sampler_path)

    def get_last_checkpoint(self) -> dict | None:
        """
        Get information about the last checkpoint.

        Returns:
            Resume info dictionary or None if no checkpoint exists
        """
        return checkpoint_utils.get_last_checkpoint(self.config.trainer.default_local_dir)

    def get_tokenizer(self):
        """Get tokenizer from training client."""
        if self.training_client is None:
            raise RuntimeError("Training client not initialized. Call initialize_async() first.")
        return self.training_client.get_tokenizer()
