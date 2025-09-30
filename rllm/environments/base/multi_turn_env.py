from abc import ABC, abstractmethod
from typing import Any

from rllm.environments.base.base_env import BaseEnv


class MultiTurnEnvironment(BaseEnv, ABC):
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(self, task: dict | None = None, max_turns: int = 3, **kwargs):
        """
        Initialize the multi-turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "questions" field
                  with a list of questions for each turn
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__()
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        self.history = []

    def reset(self, task: dict | None = None):
        # Use the provided task if available, otherwise use the default task
        if task is not None:
            self.task = task

        self.done = False
        self.current_turn = 0
        self.history = []

        return self.task, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            # For multimodal tasks, preserve task info even when done
            final_obs = self.task if isinstance(self.task, dict) else {}
            return final_obs, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    @abstractmethod
    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        Abstract method to compute the reward based on the task and action.

        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent

        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        pass

    @staticmethod
    def from_dict(env_args: dict) -> "MultiTurnEnvironment":
        raise NotImplementedError("MultiTurnEnvironment is abstract and cannot be instantiated directly. Use a concrete subclass.")
