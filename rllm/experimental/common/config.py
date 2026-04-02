from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from rllm.agents.agent import _DEFAULT_TRAJ_NAME
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
    bypass_mode: bool = True
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


class TrainingObjective(str, Enum):
    """Supported top-level training objectives in the experimental trainer."""

    RL = "rl"
    DPO = "dpo"


class DPOPairingStrategy(str, Enum):
    """Supported pairing strategies for DPO preference construction."""

    BEST_WORST = "best_worst"


@dataclass
class DPOConfig:
    """Configuration for strict, on-policy DPO pair construction."""

    beta: float = 0.1
    pairing_strategy: DPOPairingStrategy | str = DPOPairingStrategy.BEST_WORST
    min_reward_gap: float = 0.0
    drop_ties: bool = True

    def __post_init__(self) -> None:
        self.pairing_strategy = DPOPairingStrategy(self.pairing_strategy)
        if self.beta <= 0:
            raise ValueError(f"DPO beta must be positive, got {self.beta}")
        if self.min_reward_gap < 0:
            raise ValueError(f"DPO min_reward_gap must be non-negative, got {self.min_reward_gap}")


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

    objective: TrainingObjective | str = TrainingObjective.RL
    use_rllm: bool = False  # This is ignored (assumed True) for tinker backend.
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
    # Configuration for preference-pair construction in DPO mode.
    dpo: DPOConfig = field(default_factory=DPOConfig)
    # Global loss_fn override (for tinker backend; Verl uses loss_fn_map per role)
    loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro", "cross_entropy"] | None = None
    lr_schedule: Literal["linear", "cosine", "constant"] = "constant"
    warmup_steps_ratio: float = 0.0

    # Custom loss / rollout correction fields (used by Fireworks backend with cookbook losses)
    kl_beta: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    loss_agg_mode: Literal["token_mean", "seq_mean_token_sum", "seq_mean_token_mean", None] = None
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    router_replay: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "AlgorithmConfig":
        """Create an AlgorithmConfig from a dictionary configuration.

        Args:
            config: Dictionary configuration.
        Returns:
            AlgorithmConfig: The AlgorithmConfig built from the configuration.
        """
        algorithm_cfg = config.rllm.algorithm if hasattr(config, "rllm") else config.algorithm
        stepwise_cfg = config.rllm.stepwise_advantage if hasattr(config, "rllm") else config.stepwise_advantage
        dpo_cfg = algorithm_cfg.get("dpo", {})
        if isinstance(dpo_cfg, DictConfig):
            dpo_kwargs = OmegaConf.to_container(dpo_cfg, resolve=True)
        elif dpo_cfg is None:
            dpo_kwargs = {}
        else:
            dpo_kwargs = dict(dpo_cfg)
        rc_section = algorithm_cfg.get("rollout_correction", {})
        rollout_correction = RolloutCorrectionConfig(
            tis_mode=rc_section.get("tis_mode", None),
            bypass_mode=rc_section.get("bypass_mode", True),
            tis_cap=rc_section.get("tis_cap", 5.0),
        )
        return cls(
            objective=algorithm_cfg.get("objective", TrainingObjective.RL.value),
            estimator=rLLMAdvantageEstimator(algorithm_cfg.adv_estimator),
            stepwise_advantage_mode=stepwise_cfg.mode,
            norm_adv_by_std_in_grpo=algorithm_cfg.get("norm_adv_by_std_in_grpo", True),
            use_rllm=algorithm_cfg.get("use_rllm", False),
            use_precomputed_advantage=algorithm_cfg.get("use_precomputed_advantage", False),
            dpo=DPOConfig(**dpo_kwargs),
            loss_fn=algorithm_cfg.get("loss_fn", None),
            lr_schedule=algorithm_cfg.get("lr_schedule", "constant"),
            warmup_steps_ratio=algorithm_cfg.get("warmup_steps_ratio", 0.0),
            kl_beta=algorithm_cfg.get("kl_beta", 0.0),
            eps_clip=algorithm_cfg.get("eps_clip", 0.2),
            eps_clip_high=algorithm_cfg.get("eps_clip_high", None),
            loss_agg_mode=algorithm_cfg.get("loss_agg_mode", None),
            rollout_correction=rollout_correction,
            router_replay=algorithm_cfg.get("router_replay", False),
        )

    def __post_init__(self):
        self.objective = TrainingObjective(self.objective)
        if not isinstance(self.dpo, DPOConfig):
            if isinstance(self.dpo, DictConfig):
                dpo_kwargs = OmegaConf.to_container(self.dpo, resolve=True)
            else:
                dpo_kwargs = dict(self.dpo)
            self.dpo = DPOConfig(**dpo_kwargs)

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

        if self.objective == TrainingObjective.DPO and self.use_precomputed_advantage:
            raise ValueError("DPO objective cannot be combined with use_precomputed_advantage=True")
