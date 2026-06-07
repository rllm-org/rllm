"""
This module defines data structures and base classes for reward calculations
to evaluate model responses for various problem types, including math and coding.
"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RewardConfig:
    apply_format_reward: bool = False

    # Config for math-based rewards
    math_reward_weight: float = 1.0
    use_math_orm: bool = False

    # Config for code-based rewards
    code_reward_weight: float = 1.0

    # Config for cot-based rewards
    cot_reward_weight: float = 0.0

    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0

    # Bonus reward for calling tools.
    toolcall_bonus: float = 0.5

    # Toggle for using Together Code Interpreter
    use_together_code_interpreter: bool = False


class RewardType(Enum):
    """
    Enum class representing the different types of rewards that can be assigned.

    Attributes:
        MATH (str): Represents a math-related problem type.
        CODE (str): Represents a coding-related problem type.
        UNK (str): Represents an unknown or unclassified problem type.
    """

    MATH = "MATH"
    CODE = "CODE"
    WEB = "WEB"
    UNK = "UNK"


@dataclass(slots=True, kw_only=True)
class RewardInput:
    """Data structure for input required to calculate rewards.

    Attributes:
        task_info (Dict): The task dictionary containing question, answer, and other metadata
        action (str): The agent's response/solution that needs evaluation
    """

    task_info: dict
    action: str


@dataclass(slots=True, kw_only=True)
class LiveCodebenchInput:
    """Data structure for input required to calculate rewards."""

    problem_type: RewardType = RewardType.CODE
    question: str
    generation_code: str
    problem: dict
    difficult: str = "easy"


@dataclass(slots=True, kw_only=True)
class RewardOutput:
    """Data structure for the output of reward calculations.

    Attributes:
        reward (float): The computed reward value based on the evaluation of the model's response.
        metadata (dict): Additional information about the reward calculation.
        is_correct (bool): A boolean flag indicating whether the model's response is deemed correct.
    """

    reward: float
    metadata: dict = field(default_factory=dict)
    is_correct: bool | None = None
