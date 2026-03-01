"""Policy training module for Tinker-based RL.

This module handles gradient updates, model checkpointing, and data processing.
It does NOT contain any environment or agent logic.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from functools import wraps
from typing import TYPE_CHECKING, Literal

import tinker
from omegaconf import OmegaConf
from tinker.types import AdamParams
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.tokenizer_utils import Tokenizer

from rllm.agents.agent import TrajectoryGroup
from rllm.experimental.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    TransformConfig,
    rLLMAdvantageEstimator,
)
from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Mapping from rLLMAdvantageEstimator to their default Tinker loss function (overriding is allowed through config)
ADV_TO_LOSS_FN_AUTO_MAP = {
    rLLMAdvantageEstimator.REINFORCE: "importance_sampling",
    rLLMAdvantageEstimator.GRPO: "ppo",
    rLLMAdvantageEstimator.RLOO: "importance_sampling",
    rLLMAdvantageEstimator.OTHER: "importance_sampling",
}

DEFAULT_LOSS_FN = "importance_sampling"


# helper decorator for any function requiring a training client to be initialized
def require_training_client(func):
    def _check_training_client(self):
        if self.training_client is None:
            raise RuntimeError("Training client not initialized. Call initialize_async() first.")

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            _check_training_client(self)
            return await func(self, *args, **kwargs)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            _check_training_client(self)
            return func(self, *args, **kwargs)

        return sync_wrapper


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
        cf_config: CompactFilteringConfig | None = None,
        transform_config: TransformConfig | None = None,
        algorithm_config: AlgorithmConfig | None = None,
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
        # fill in the default versions of the configs if not provided
        self.cf_config = cf_config or CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)
        self.transform_config = transform_config or TransformConfig()
        self.algorithm_config = algorithm_config or AlgorithmConfig.from_config(self.config)

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
            tinker_model_id = OmegaConf.select(self.config, "training.resume_from_tinker_id", default=None)
            if tinker_model_id and tinker_model_id.startswith("tinker://") and "/weights/" in tinker_model_id:
                checkpoint_name = tinker_model_id.split("/weights/")[-1]
                try:
                    batch = int(checkpoint_name)
                except ValueError:
                    batch = 0

                resume_info = {
                    "state_path": tinker_model_id,
                    "sampler_path": tinker_model_id.replace("/weights/", "/sampler_weights/"),
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

            start_batch = resume_info["batch"]
            logger.info(f"Resuming from batch {start_batch}")
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

    @require_training_client
    async def _get_forward_backward_futures(
        self,
        training_datums: list[tinker.Datum] | dict[str, list[tinker.Datum]],
        estimator_map: dict[str, rLLMAdvantageEstimator | str],
        algorithm_config: AlgorithmConfig,
    ) -> list[tinker.APIFuture]:
        fwd_bwd_futures = []
        if isinstance(training_datums, dict):
            for group_role, datums in training_datums.items():
                estimator = estimator_map.get(group_role, self.algorithm_config.estimator)
                loss_fn = algorithm_config.loss_fn or ADV_TO_LOSS_FN_AUTO_MAP.get(estimator, DEFAULT_LOSS_FN)
                fwd_bwd_future = await self.training_client.forward_backward_async(
                    [self._remove_mask(datum) for datum in datums],
                    loss_fn=loss_fn,  # type: ignore[attr-defined]
                )
                fwd_bwd_futures.append(fwd_bwd_future)
        else:
            loss_fn = algorithm_config.loss_fn or ADV_TO_LOSS_FN_AUTO_MAP.get(algorithm_config.estimator, DEFAULT_LOSS_FN)
            fwd_bwd_future = await self.training_client.forward_backward_async(
                [self._remove_mask(datum) for datum in training_datums],
                loss_fn=loss_fn,  # type: ignore[attr-defined]
            )
            fwd_bwd_futures.append(fwd_bwd_future)

        return fwd_bwd_futures

    @require_training_client
    async def forward_backward_from_trajectory_groups(
        self,
        trajectory_groups: list[TrajectoryGroup],
        algorithm_config: AlgorithmConfig | None = None,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict]:
        """
        Run forward-backward pass from trajectory groups (skipping episode transformation).

        This method is useful when trajectory groups have already been computed by the
        unified trainer pipeline.

        Args:
            trajectory_groups: List of TrajectoryGroup objects (already filtered/transformed)
            algorithm_config: Algorithm config for advantage computation (uses self.algorithm_config if None)

        Returns:
            Tuple of (training_datums, training_logprobs, adv_metrics)
            - training_datums: List of datums WITH masks for metrics (or dictionary of datums, keyed by trajectory group role)
            - training_logprobs: List of training logprobs from forward-backward
            - adv_metrics: Dictionary of advantage metrics (rewards + advantages summary for each trajectory group)
        """
        if algorithm_config is None:
            algorithm_config = self.algorithm_config

        # Transform trajectory groups to datums (includes advantage computation)
        training_datums, adv_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=algorithm_config,
        )

        # Forward-backward pass
        fwd_bwd_futures = await self._get_forward_backward_futures(
            training_datums=training_datums,
            estimator_map=algorithm_config.estimator_map,
            algorithm_config=algorithm_config,
        )

        # Wait for completion and extract logprobs
        fwd_bwd_results = await asyncio.gather(*fwd_bwd_futures)

        # Extract training logprobs from loss_fn_outputs
        training_logprobs = []
        for fwd_bwd_result in fwd_bwd_results:
            for output in fwd_bwd_result.loss_fn_outputs:
                logprobs = output["logprobs"].to_torch()
                training_logprobs.append(logprobs)

        return training_datums, training_logprobs, adv_metrics

    @require_training_client
    async def optim_step_future(
        self,
        step: int,
        total_steps: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> tuple[tinker.APIFuture[tinker.types.OptimStepResponse], float]:
        scheduled_learning_rate = learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=self.algorithm_config.lr_schedule,
            warmup_steps_ratio=self.algorithm_config.warmup_steps_ratio,
            step=step,
            total_steps=total_steps,
        )

        adam_params = AdamParams(
            learning_rate=scheduled_learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        optim_step_future = await self.training_client.optim_step_async(adam_params)  # type: ignore[attr-defined]
        return optim_step_future, scheduled_learning_rate

    @require_training_client
    async def fused_forward_backward_and_optim_step(
        self,
        step: int,
        total_steps: int,
        trajectory_groups: list[TrajectoryGroup],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], list[torch.Tensor], dict, float]:
        """Run forward-backward pass and optimizer step from trajectory groups -- at the same time.

        This follows from the best-practice with Tinker: https://tinker-docs.thinkingmachines.ai/async#performance-tips-overlap-requests
        """
        # Transform trajectory groups to datums (includes advantage computation)
        training_datums, adv_metrics = transform_trajectory_groups_to_datums(
            trajectory_groups,
            algorithm_config=self.algorithm_config,
        )

        # Forward-backward and optimizer future together
        fwd_bwd_futures = await self._get_forward_backward_futures(
            training_datums=training_datums,
            estimator_map=self.algorithm_config.estimator_map,
            algorithm_config=self.algorithm_config,
        )

        optim_step_future, scheduled_learning_rate = await self.optim_step_future(
            step=step,
            total_steps=total_steps,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        # Retrieve the results together
        fwd_bwd_results = await asyncio.gather(*fwd_bwd_futures)
        await optim_step_future.result_async()

        training_logprobs = []
        for fwd_bwd_result in fwd_bwd_results:
            for output in fwd_bwd_result.loss_fn_outputs:
                logprobs = output["logprobs"].to_torch()
                training_logprobs.append(logprobs)

        return training_datums, training_logprobs, adv_metrics, scheduled_learning_rate

    @require_training_client
    async def save_checkpoint_and_get_sampling_client(
        self,
        batch_idx: int,
        kind: str = "both",
        do_save: bool = False,
    ) -> tinker.SamplingClient:
        """
        Save checkpoint and return paths.

        Args:
            batch_idx: Current batch index
            kind: Checkpoint kind ("state", "sampler", or "both")
            do_save: Whether to save the checkpoint

        Returns:
            Dictionary with checkpoint paths
        """
        if do_save:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=self.training_client,
                name=f"{batch_idx:06d}",
                log_path=self.config.training.default_local_dir,
                kind=kind,
                loop_state={"batch": batch_idx},
            )
            return self.training_client.create_sampling_client(path_dict["sampler_path"])
        else:
            return await self.training_client.save_weights_and_get_sampling_client_async()

    @require_training_client
    def create_sampling_client(self, sampler_path: str) -> tinker.SamplingClient:
        """
        Create a sampling client from a checkpoint path.

        Args:
            sampler_path: Path to sampler checkpoint

        Returns:
            Tinker sampling client
        """
        return self.training_client.create_sampling_client(sampler_path)  # type: ignore[attr-defined]

    def get_last_checkpoint(self) -> dict | None:
        """
        Get information about the last checkpoint.

        Returns:
            Resume info dictionary or None if no checkpoint exists
        """
        return checkpoint_utils.get_last_checkpoint(self.config.training.default_local_dir)

    @require_training_client
    def get_tokenizer(self) -> Tokenizer:
        """Get tokenizer from training client."""
        return self.training_client.get_tokenizer()  # type: ignore[attr-defined]


"""
Learning rate scheduler for Tinker. Add warmup steps support.
Adapted from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/utils/lr_scheduling.py
"""

LRSchedule = Literal["linear", "cosine", "constant"]


def compute_schedule_lr_multiplier(lr_schedule: LRSchedule, warmup_steps_ratio: float, step: int, total_steps: int) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule

    Args:
        lr_schedule: Learning rate schedule
        warmup_steps_ratio: Ratio of warmup steps to total steps
        step: Current step
        total_steps: Total steps

    Returns:
        Learning rate multiplier
    """
    import math

    warmup_steps = int(total_steps * warmup_steps_ratio)
    if step < warmup_steps:
        return step / warmup_steps
    # Adjust step and total_steps for warmup steps
    step, total_steps = step - warmup_steps, total_steps - warmup_steps
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif lr_schedule == "constant":
        return 1
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
