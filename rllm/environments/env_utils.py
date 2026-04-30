import concurrent.futures
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from rllm.types import Trajectory


def compute_trajectory_reward(trajectory: "Trajectory") -> "Trajectory":
    """
    Add trajectory reward to the dict of each interaction.

    Args:
        trajectory: List of dictionaries representing each step in the trajectory.

    Returns:
        The updated trajectory with trajectory_reward added to each step.
    """
    if not trajectory:
        return trajectory
    trajectory_reward = np.sum([d.reward for d in trajectory.steps])
    trajectory.reward = trajectory_reward
    return trajectory


def compute_mc_return(trajectory: "Trajectory", gamma: float = 0.95) -> "Trajectory":
    """
    In-place Monte Carlo returns for a Trajectory dataclass.

    G_t = R_{t+1} + γ * G_{t+1}

    Args:
        trajectory: Trajectory object whose .steps is a list of Step objects.
        gamma: Discount factor.

    Returns:
        The same Trajectory, with each step.mc_return filled.
    """
    G = 0.0
    # Walk backward through the list of Step objects
    for step in reversed(trajectory.steps):
        # step.reward is R_{t+1} by your definition
        G = step.reward + gamma * G
        step.mc_return = G
    return trajectory


@contextmanager
def parallel_task_manager(func: Callable, items: list[Any], max_workers: int = 32) -> Iterator[list[tuple[int, Any]]]:
    """
    Execute a function in parallel for all items and collect results.

    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers

    Yields:
        List of (idx, result) tuples
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, *item): i for i, item in enumerate(items)}
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            result = future.result()
            results.append((idx, result))
    yield results
