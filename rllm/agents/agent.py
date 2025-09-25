from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)

    observation: Any = None
    thought: str = ""
    action: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)  # Store any additional info.

    # field below are filled by the engine
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

    step_id: str = ""
    step_num: int = 0


@dataclass
class Action:
    action: Any = None


@dataclass
class Trajectory:
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self):
        return {
            "task": self.task,
            "steps": [asdict(step) for step in self.steps],
            "reward": float(self.reward),
        }


@dataclass
class Episode:
    id: str = ""
    task: Any = None
    termination_reason = None
    is_correct: bool = False
    trajectories: list[tuple[str, Trajectory]] = field(default_factory=list)  # [(agent_name, Trajectory), ...]
    metrics: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "task": self.task,
            "termination_reason": self.termination_reason.value if self.termination_reason is not None else None,
            "is_correct": bool(self.is_correct),
            "trajectories": [(agent_name, trajectory.to_dict()) for agent_name, trajectory in self.trajectories],
            "metrics": self.metrics,
            "meta": self.meta,
        }


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Converts agent's internal state into a list of OAI chat completions."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Converts agent's internal state into a Trajectory object."""
        return Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation (Any): The observation after stepping through environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended due to termination.
            info (dict): Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement this method if using AgentExecutionEngine")

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method if using AgentExecutionEngine")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Step | None:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
