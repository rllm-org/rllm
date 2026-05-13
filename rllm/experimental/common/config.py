from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from rllm.types import _DEFAULT_TRAJ_NAME
from rllm.workflows.workflow import TerminationReason


@dataclass
class AsyncTrainingConfig:
    """Controls the async training behavior spectrum.

    When `enable` is False, the trainer uses the current synchronous pipeline.
    When `enable` is True, the trainer runs concurrent generation + training
    with group-level streaming and dispatch-time throttle.

    Behavior spectrum:
        - staleness_threshold=0, trigger_parameter_sync_step=1: On-policy
        - staleness_threshold=0, trigger_parameter_sync_step=K: Stream off-policy
        - staleness_threshold>0, partial_rollout=False: Async with staleness
        - staleness_threshold>0, partial_rollout=True: Async with partial rollout
    """

    enable: bool = False
    mini_batch_size: int = 1  # episode groups per optimizer step
    fwd_bwd_group_size: int | None = None  # task batches per forward-backward pass (default: mini_batch_size)
    staleness_threshold: float = 0.0  # 0.0 = on-policy. Controls dispatch throttle quota.
    trigger_parameter_sync_step: int = 1  # optimizer steps between weight sync + version bump
    partial_rollout: bool = True  # enable turn-level gating during weight sync
    episode_offload_dir: str | None = None  # NVMe offload dir for pending episodes (None = disabled)
    trajectory_group_offload_dir: str | None = None  # NVMe offload dir for queued task batches (None = disabled)

    def __post_init__(self):
        if self.fwd_bwd_group_size is None:
            self.fwd_bwd_group_size = self.mini_batch_size
        if self.enable:
            assert self.fwd_bwd_group_size >= 1
            assert self.mini_batch_size % self.fwd_bwd_group_size == 0, f"mini_batch_size ({self.mini_batch_size}) must be divisible by fwd_bwd_group_size ({self.fwd_bwd_group_size})"

    @classmethod
    def from_config(cls, config: DictConfig) -> "AsyncTrainingConfig":
        return cls(**OmegaConf.to_container(config))  # type: ignore


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
        return (
            (self.mask_max_prompt_length_exceeded and termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)
            or (self.mask_max_response_length_exceeded and termination_reason == TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
            or (self.mask_env_done and termination_reason == TerminationReason.ENV_DONE)
            or (self.mask_max_turns_exceeded and termination_reason == TerminationReason.MAX_TURNS_EXCEEDED)
            or (self.mask_timeout and termination_reason == TerminationReason.TIMEOUT)
            or (self.mask_unknown and termination_reason == TerminationReason.UNKNOWN)
            or (self.mask_error and termination_reason == TerminationReason.ERROR)
        )


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

    @classmethod
    def from_config(cls, transform_config: DictConfig, *, broadcast: bool = True) -> "TransformConfig":
        return cls(
            impute_missing_names=transform_config.get("impute_missing_names", True),
            default_traj_name=transform_config.get("default_traj_name", _DEFAULT_TRAJ_NAME),
            drop_unnamed_traj=transform_config.get("drop_unnamed_traj", False),
            broadcast=broadcast,
        )


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

    # Filter out episode groups where all rollouts have the same is_correct (no gradient signal).
    # Applied at the accumulator level in async training, before groups enter the buffer.
    filter_uniform_groups: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "RejectionSamplingConfig":
        mode = config.get("mode", None)
        if mode is None:
            mode = "episode" if config.get("enable", False) else "none"
        return cls(
            mode=mode,
            min_trajs_per_group=config.get("min_trajs_per_group", 2),
            min_partial_solve_tasks=config.get("min_partial_solve_tasks", 1),
            filter_uniform_groups=config.get("filter_uniform_groups", False),
        )


@dataclass
class RolloutCorrectionConfig:
    """Configuration for rollout correction (TIS, proximal forward passes).

    Backend-agnostic — each backend interprets these according to its infrastructure.

    Attributes:
        tis_mode: None = disabled (string loss names, current behavior).
              "token" or "sequence" = enable custom callable loss with TIS at that level.
        bypass_mode: When True, use rollout (inference) logprobs as π_old — no
              proximal forward pass. When False, compute π_old via policy.forward()
              (3-policy / decoupled PPO).
        tis_cap: Upper clamp on the TIS importance weight.
    """

    tis_mode: str | None = None
    bypass_mode: bool | None = None
    tis_cap: float = 5.0


class rLLMAdvantageEstimator(str, Enum):
    """
    A unified advantage estimator for rLLM. Work with both `tinker` and `verl` backends at the expense of
    losing some flexibility.
    TODO(listar2000): add more estimators.
    """

    GRPO = "grpo"
    REINFORCE = "reinforce"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    RLOO = "rloo"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> "rLLMAdvantageEstimator":
        return cls.OTHER


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters.

    ``estimator_map`` values may be:

    * A bare estimator name or enum — only the advantage estimator is overridden.
    * A ``(estimator, policy_loss)`` tuple — both the advantage estimator *and*
      the backend policy loss function are overridden for that role.

    During ``__post_init__``, tuples are split: the estimator goes into
    ``estimator_map`` and the loss function goes into ``loss_fn_map``.
    """

    use_rllm: bool | None = None  # Deprecated
    estimator: rLLMAdvantageEstimator = rLLMAdvantageEstimator.GRPO
    estimator_map: dict[str, rLLMAdvantageEstimator | str | tuple] = field(default_factory=dict)
    # Per-role policy loss overrides (populated from tuples in estimator_map during __post_init__)
    loss_fn_map: dict[str, str] = field(default_factory=dict)
    # TODO(listar2000): eventually we will remove the `per_step` mode all-together. Now we keep it for backward compatibility.
    stepwise_advantage_mode: Literal["broadcast", "per_step"] = "broadcast"
    norm_adv_by_std_in_grpo: bool = True
    # When True, always use pre-computed step.advantage from the workflow and skip
    # advantage computation (GRPO/REINFORCE). Steps missing advantages default to 0.0.
    # When False (default), always compute advantages normally.
    use_precomputed_advantage: bool = False
    # Global loss function (backend-specific values; null = backend default)
    loss_fn: str | None = None
    lr_schedule: Literal["linear", "cosine", "constant"] = "constant"
    warmup_steps: int = -1
    warmup_steps_ratio: float = 0.0

    # Custom loss / rollout correction fields (used by Fireworks backend with cookbook losses)
    kl_beta: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", None] = None
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    router_replay: Literal["disabled", "R2", "R3"] = "disabled"

    @classmethod
    def from_config(cls, algorithm_config: DictConfig, *, stepwise_advantage_mode: str = "broadcast", estimator_map: dict | None = None) -> "AlgorithmConfig":
        """Create an AlgorithmConfig from a dictionary configuration.

        Args:
            algorithm_config: Dictionary configuration.
        Returns:
            AlgorithmConfig: The AlgorithmConfig built from the configuration.
        """
        if algorithm_config.get("use_rllm", None) is not None:
            from warnings import warn

            warn(
                "`algorithm.use_rllm` is deprecated and ignored — advantages are always computed via the rLLM-native path. Remove the field from your config.",
                DeprecationWarning,
                stacklevel=2,
            )

        rc_section = algorithm_config.get("rollout_correction", {})
        rollout_correction = RolloutCorrectionConfig(
            tis_mode=rc_section.get("tis_mode", None),
            bypass_mode=rc_section.get("bypass_mode", None),
            tis_cap=rc_section.get("tis_cap", 2.0),
        )
        return cls(
            estimator=rLLMAdvantageEstimator(algorithm_config.adv_estimator),
            estimator_map=estimator_map or {},
            stepwise_advantage_mode=stepwise_advantage_mode,
            norm_adv_by_std_in_grpo=algorithm_config.get("norm_adv_by_std_in_grpo", True),
            use_precomputed_advantage=algorithm_config.get("use_precomputed_advantage", False),
            loss_fn=algorithm_config.get("loss_fn", None),
            lr_schedule=algorithm_config.get("lr_schedule", "constant"),
            warmup_steps=algorithm_config.get("warmup_steps", -1),
            warmup_steps_ratio=algorithm_config.get("warmup_steps_ratio", 0.0),
            kl_beta=algorithm_config.get("kl_beta", 0.0),
            eps_clip=algorithm_config.get("eps_clip", 0.2),
            eps_clip_high=algorithm_config.get("eps_clip_high", None),
            loss_agg_mode=algorithm_config.get("loss_agg_mode", None),
            rollout_correction=rollout_correction,
            router_replay=algorithm_config.get("router_replay", "disabled"),
        )

    def __post_init__(self):
        if self.use_rllm is not None:
            from warnings import warn

            warn(
                "`algorithm.use_rllm` is deprecated and ignored — advantages are always computed via the rLLM-native path. Remove the field from your config.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Normalize estimator_map: split (estimator, loss_fn) tuples.
        normalized_map: dict[str, rLLMAdvantageEstimator | str] = {}
        for role, value in self.estimator_map.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"estimator_map tuple for role '{role}' must have exactly 2 elements (estimator, loss_fn), got {len(value)}")
                estimator, loss_fn = value
                normalized_map[role] = estimator
                self.loss_fn_map[role] = str(loss_fn)
            else:
                normalized_map[role] = value
        self.estimator_map = normalized_map

        if self.stepwise_advantage_mode == "per_step":
            from warnings import warn

            warn(
                "The `per_step` mode is deprecated in experimental unified trainer. "
                "Set to `broadcast` mode automatically. Please either use the legacy "
                "trainers (`agent_workflow_trainer` for `Verl` or "
                "`tinker_workflow_trainer` for `Tinker`) with the `per_step` "
                "configuration, or manually pass in a hook with the implementation "
                "of `per_step` advantage computation logic.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.stepwise_advantage_mode = "broadcast"
