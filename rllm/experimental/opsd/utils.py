from dataclasses import dataclass


@dataclass
class OPSDConfig:
    """Configuration for (on-policy) self-distillation."""

    kl_penalty_coef: float
    kl_discount_factor: float
    teacher_messages_key: str = "teacher_messages"
    teacher_policy_update_freq: int = 1  # -1 for always using the initial policy
