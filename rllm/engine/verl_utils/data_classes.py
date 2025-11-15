from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rllm.workflows.workflow import TerminationReason


@dataclass
class CompactFilteringConfig:
    """Configuration for compact filtering of episodes based on termination reasons.

    Compatible with OmegaConf/Hydra config system.
    All fields default to False for backwards compatibility.

    Usage with OmegaConf:
        config = OmegaConf.structured(CompactFilteringConfig)
        # or from YAML
        config = OmegaConf.load("config.yaml").rllm.compact_filtering
    """

    enable: bool = False
    mask_max_prompt_length_exceeded: bool = False
    mask_max_response_length_exceeded: bool = False
    mask_env_done: bool = False
    mask_max_turns_exceeded: bool = False
    mask_timeout: bool = False
    mask_unknown: bool = False
    mask_error: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "CompactFilteringConfig":
        """Create a CompactFilteringConfig from a dictionary configuration.

        Args:
            config: Dictionary configuration.
        Returns:
            CompactFilteringConfig: The CompactFilteringConfig built from the configuration.
        """
        if isinstance(config, dict):
            return cls(**config)
        elif isinstance(config, CompactFilteringConfig):
            return config
        else:
            raise ValueError(f"Invalid configuration type: {type(config)}")

    def should_mask(self, termination_reason: TerminationReason) -> bool:
        """Check if a specific termination reason should be masked/filtered out.

        Args:
            termination_reason: The termination reason to check.
        Returns:
            True if this termination reason should be filtered out, False otherwise.
        """
        if not self.enable:
            return False
        return (self.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED) or (self.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED) or (self.mask_env_done and termination_reason == TerminationReason.ENV_DONE) or (self.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED) or (self.mask_timeout and termination_reason == TerminationReason.TIMEOUT) or (self.mask_unknown and termination_reason == TerminationReason.UNKNOWN) or (self.mask_error and termination_reason == TerminationReason.ERROR)


@dataclass
class ProcessedStepData:
    """Represents one tokenized step/trajectory unit ready for batching.

    This is the atomic unit that will become one row in the final batch.
    In non-stepwise mode, this represents an entire trajectory.
    In stepwise mode, this represents a single step within a trajectory.
    """

    prompt: torch.Tensor
    response: torch.Tensor
    mask: torch.Tensor
    step_reward: float
    step_id: str


@dataclass
class AccumulatedData:
    """Container for all accumulated lists during episode processing.

    Each list has one entry per ProcessedStepData (i.e., per batch row).
    """

    # Tensor data (to be batched)
    prompts: list[torch.Tensor] = field(default_factory=list)
    responses: list[torch.Tensor] = field(default_factory=list)
    traj_mask: list[torch.Tensor] = field(default_factory=list)

    # Reward data (parallel to tensor lists)
    traj_rewards: list[float] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)

    # ID tracking (parallel to tensor lists)
    episode_ids: list[str] = field(default_factory=list)
    trajectory_ids: list[str] = field(default_factory=list)
    step_ids: list[str] = field(default_factory=list)

    # Metadata (parallel to tensor lists)
    step_nums: list[int] = field(default_factory=list)  # number of steps in the trajectory
    is_last_step: list[bool] = field(default_factory=list)
    is_correct: list[bool] = field(default_factory=list)
    termination_reasons: list[Any] = field(default_factory=list)  # TerminationReason enum values
    metrics: list[dict] = field(default_factory=list)

    # Episode-level tracking
    repeat_counts: list[int] = field(default_factory=list)  # number of batch rows per episode

    def add_step(self, step_data: ProcessedStepData, episode_id: str, trajectory_id: str, traj_reward: float, step_num: int, is_last: bool, is_correct: bool, termination_reason: Any, metrics: dict):
        """Add a single processed step to all accumulator lists.

        This helper ensures all lists stay in sync and reduces boilerplate.
        """
        self.prompts.append(step_data.prompt)
        self.responses.append(step_data.response)
        self.traj_mask.append(step_data.mask)
        self.step_rewards.append(step_data.step_reward)
        self.traj_rewards.append(traj_reward)
        self.step_ids.append(step_data.step_id)

        self.episode_ids.append(episode_id)
        self.trajectory_ids.append(trajectory_id)
        self.step_nums.append(step_num)
        self.is_last_step.append(is_last)
        self.is_correct.append(is_correct)
        self.termination_reasons.append(termination_reason)
        self.metrics.append(metrics)

    def __len__(self) -> int:
        """Return the total number of batch rows accumulated."""
        return len(self.prompts)


@dataclass
class BatchedTensors:
    """Container for batched and padded tensors ready for DataProto.

    Separates the batching logic output from the DataProto construction.
    """

    # Main sequence data
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    # Separated prompt/response views
    prompts: torch.Tensor
    responses: torch.Tensor
    response_mask: torch.Tensor  # trajectory mask

    # Reward tensors (aligned with response tokens)
    traj_rewards: torch.Tensor
    step_rewards: torch.Tensor

    # Non-tensor metadata (to be passed through)
    episode_ids: np.ndarray
    trajectory_ids: np.ndarray
    step_ids: np.ndarray
    step_nums: np.ndarray
    is_correct: np.ndarray
    termination_reasons: np.ndarray
    metrics: np.ndarray
    is_last_step: np.ndarray
    is_valid: np.ndarray  # filtering flags

    # Meta info
    repeat_counts: list[int]
