from dataclasses import dataclass
from enum import Enum
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from rllm.agents.agent import _DEFAULT_TRAJ_NAME
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
    def from_config(cls, config: DictConfig) -> "CompactFilteringConfig":
        """Create a CompactFilteringConfig from a dictionary configuration.

        Args:
            config: Dictionary configuration.
        Returns:
            CompactFilteringConfig: The CompactFilteringConfig built from the configuration.
        """
        return cls(**OmegaConf.to_container(config))  # type: ignore

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
class TransformConfig:
    """Configuration for the episode-to-group transformation pipeline."""

    # Name imputation
    impute_missing_names: bool = True
    # Default trajectory name (if user does not provide a name); treated as unnamed
    default_traj_name: str = _DEFAULT_TRAJ_NAME
    # Whether to drop unnamed trajectories on failure
    drop_unnamed_traj: bool = False

    # Reward configuration
    broadcast: bool = True  # If True, use trajectory-level rewards; if False, use per-step rewards


@dataclass
class RejectionSamplingConfig:
    """Configuration for rejection sampling."""

    # Rejection sampling mode
    # - "none": No rejection, just track metrics
    # - "episode": Skip whole batch if criteria not met, accumulate until enough partial solves
    # - "group": Filter groups with insufficient trajectories, pass remaining to trainer
    mode: Literal["none", "episode", "group"] = "none"

    # Minimum trajectories required per trajectory group (for "group" mode)
    min_trajs_per_group: int = 2

    # For "episode" mode (verl compatibility): minimum number of tasks with partial solves before proceeding
    min_partial_solve_tasks: int = 1


class rLLMAdvantageEstimator(str, Enum):
    """
    A unified advantage estimator for rLLM. Work with both `tinker` and `verl` backends at the expense of
    losing some flexibility. Currently only supporting GRPO and REINFORCE.
    TODO(listar2000): add more estimators.
    """

    GRPO = "grpo"
    REINFORCE = "reinforce"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> "rLLMAdvantageEstimator":
        return cls.OTHER


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters."""

    estimator: rLLMAdvantageEstimator = rLLMAdvantageEstimator.GRPO
    stepwise_advantage_mode: Literal["broadcast", "per_step"] = "broadcast"
    norm_adv_by_std_in_grpo: bool = True
    use_rllm: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "AlgorithmConfig":
        """Create an AlgorithmConfig from a dictionary configuration.

        Args:
            config: Dictionary configuration.
        Returns:
            AlgorithmConfig: The AlgorithmConfig built from the configuration.
        """
        return cls(
            estimator=rLLMAdvantageEstimator(config.algorithm.adv_estimator),
            stepwise_advantage_mode=config.rllm.stepwise_advantage.mode,
            norm_adv_by_std_in_grpo=config.rllm.stepwise_advantage.get("norm_adv_by_std_in_grpo", True),
            use_rllm=config.rllm.stepwise_advantage.get("use_rllm", False),
        )
