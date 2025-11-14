import re
import string
from collections import Counter
from typing import Protocol, runtime_checkable

from rllm.agents.agent import Action
from rllm.rewards.code_reward import RewardCodeFn
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example implementation
def zero_reward(task_info: dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task: The task dictionary
        action: The agent's response

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for search tasks that implements the RewardFunction protocol.

    Args:
        task_info: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value based on search evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardSearchFn(reward_config)
    if isinstance(action, Action):
        action = action.action

    # Create RewardInput from task_info and action
    reward_input = RewardInput(task_info=task_info, action=action)

    return reward_fn(reward_input)


def code_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for code tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on code execution results
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def f1_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function that computes F1 score between predicted text and gold text.

    This function normalizes both texts (lowercase, remove punctuation, remove articles,
    fix whitespace) before tokenizing and computing F1 score based on token overlap.

    Args:
        task_info: The task dictionary containing ground_truth (gold text)
        action: The agent's predicted response/solution

    Returns:
        RewardOutput: The calculated reward value (F1 score)

    Example:
        >>> task_info = {"ground_truth": "Hello, world!"}
        >>> action = "hello there world"
        >>> output = f1_reward_fn(task_info, action)
        >>> print(output.reward)  # F1 score between the texts
    """
    if isinstance(action, Action):
        action = action.action

    # Extract gold text from task_info
    gold_text = task_info.get("ground_truth", "")
    if gold_text is None:
        gold_text = ""

    def normalize_text(s: str) -> str:
        """Normalize text for evaluation (following HotpotQA/SQuAD standards)"""
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # Normalize and tokenize both texts
    predicted_normalized = normalize_text(str(action))
    gold_normalized = normalize_text(str(gold_text))
    predicted_tokens = predicted_normalized.split()
    gold_tokens = gold_normalized.split()

    # Handle empty cases - if neither predicted nor gold are passed, return 0
    if not predicted_tokens and not gold_tokens:
        f1_score = 0.0
    elif not predicted_tokens or not gold_tokens:
        f1_score = 0.0
    else:
        # Compute token overlap using Counter intersection
        predicted_counter = Counter(predicted_tokens)
        gold_counter = Counter(gold_tokens)
        common = predicted_counter & gold_counter
        num_same = sum(common.values())

        if num_same == 0:
            f1_score = 0.0
        else:
            precision = num_same / len(predicted_tokens)
            recall = num_same / len(gold_tokens)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return RewardOutput(reward=f1_score, metadata={})
