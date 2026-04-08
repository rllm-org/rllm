"""Countdown-specific utilities for the GSD experiment.

Provides dataset loading, reward wiring, seed hints, and prompt builders
for the countdown number puzzle task.
"""

from __future__ import annotations

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def prepare_countdown_datasets():
    """Load pre-registered countdown train + test datasets.

    Raises:
        RuntimeError: If datasets are not registered.  Run
            ``python -m examples.countdown.prepare_countdown_data`` first.
    """
    train = DatasetRegistry.load_dataset("countdown", "train")
    test = DatasetRegistry.load_dataset("countdown", "test")
    if train is None or test is None:
        raise RuntimeError("Datasets not found. Run:\n  python -m examples.countdown.prepare_countdown_data\nto register the countdown datasets.")
    return train, test


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def countdown_reward(task: dict, response_text: str) -> float:
    """Binary countdown reward: 1.0 if correct, 0.0 otherwise."""
    result = countdown_reward_fn(task, response_text)
    return 1.0 if result.is_correct else 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COUNTDOWN_SOLVER_PROMPT = """
You are a number puzzle solver. You are given several numbers and a target. \
Find an arithmetic expression using the given numbers that equals the target. \
Use each number exactly once. You may use +, -, *, / and parentheses. \
Show your reasoning, then output your final answer in <answer>...</answer> tags, \
for example <answer> (1 + 2) / 3 </answer>.
""".strip()


def build_countdown_student_prompt(question: str) -> list[dict]:
    """Student prompt — solver system prompt + problem (no hint)."""
    return [
        {"role": "system", "content": COUNTDOWN_SOLVER_PROMPT},
        {"role": "user", "content": question},
    ]


def build_countdown_teacher_prompt(question: str, hint: str) -> list[dict]:
    """Teacher prompt — solver system prompt + hint + problem."""
    return [
        {
            "role": "system",
            "content": f"{COUNTDOWN_SOLVER_PROMPT}\n\nStrategy hint:\n{hint}",
        },
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Seed hints for the HintPool
# ---------------------------------------------------------------------------

COUNTDOWN_SEED_HINTS = [
    ("- Try working backwards from the target number.\n- Look for pairs of numbers whose product or sum is close to the target.\n- Consider whether division can simplify the problem."),
    ("- Check if the target is achievable by combining just two numbers first.\n- Use the third number to adjust the result.\n- Subtraction and division are often overlooked."),
    (
        "- Factor the target number and see if any available numbers are factors.\n"
        "- Try all pairwise operations mentally before committing.\n"
        "- Parentheses can change the order of operations significantly."
    ),
]
