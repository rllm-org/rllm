import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    chat_completions: list[dict[str, str]] = field(default_factory=list)

    observation: Any = None
    thought: str = ""
    action: Any = None
    model_response: str = ""
    model_output: "ModelOutput" = None  # noqa: F821
    info: dict = field(default_factory=dict)  # Store any additional info.

    # field below are filled by the engine
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "logprobs": self.logprobs,
            "chat_completions": self.chat_completions,
            "observation": self.observation,
            "thought": self.thought,
            "action": self.action.action if isinstance(self.action, Action) else self.action,
            "model_response": self.model_response,
            "model_output": self.model_output.to_dict() if self.model_output is not None else None,
            "info": self.info,
            "reward": self.reward,
            "done": self.done,
            "mc_return": self.mc_return,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        from rllm.engine.rollout import ModelOutput

        return cls(
            prompt_ids=data["prompt_ids"],
            response_ids=data["response_ids"],
            logprobs=data["logprobs"],
            chat_completions=data["chat_completions"],
            observation=data["observation"],
            thought=data["thought"],
            action=data["action"],
            model_response=data["model_response"],
            model_output=ModelOutput.from_dict(data["model_output"]) if data.get("model_output", None) is not None else None,
            info=data.get("info", {}),
            reward=data["reward"],
            done=data["done"],
            mc_return=data["mc_return"],
        )


@dataclass
class Action:
    action: Any = None


@dataclass
class Trajectory:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))  # unique id to deduplicate on
    name: str = "agent"
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    info: dict = field(default_factory=dict)

    def to_dict(self):
        # Remove large/non-serializable payloads (e.g., images) from task
        def _sanitize_task(task_obj):
            if isinstance(task_obj, dict):
                cleaned = {k: v for k, v in task_obj.items() if k not in ("image", "images")}
                return cleaned
            return task_obj

        return {
            "uid": self.uid,
            "name": self.name,
            "task": _sanitize_task(self.task),
            "steps": [step.to_dict() for step in self.steps],
            "reward": float(self.reward),
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Create Trajectory from dictionary, properly deserializing Step objects."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data["name"],
            task=data["task"],
            steps=[Step.from_dict(step_data) for step_data in data.get("steps", [])],
            reward=data["reward"],
            info=data.get("info", {}),
        )

    def is_cumulative(self) -> bool:
        """
        Returns True if for every step after the first, its chat_completions is an exact superset
        of the previous step's chat_completions (i.e., the previous chat_completions is a prefix).
        """
        prev = None
        for step in self.steps:
            if prev is not None:
                prev_cc = prev.chat_completions
                curr_cc = step.chat_completions
                if not (len(curr_cc) >= len(prev_cc) and curr_cc[: len(prev_cc)] == prev_cc):
                    return False
            prev = step
        return True


@dataclass
class Episode:
    id: str = ""  # rollout id e.g., task_id:rollout_idx
    task: Any = None
    termination_reason: "TerminationReason" = None  # noqa: F821
    is_correct: bool = False
    trajectories: list[Trajectory] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def to_dict(self):
        # Remove large/non-serializable payloads (e.g., images) from task
        def _sanitize_task(task_obj):
            if isinstance(task_obj, dict):
                cleaned = {k: v for k, v in task_obj.items() if k not in ("image", "images")}
                return cleaned
            return task_obj

        return {
            "id": self.id,
            "task": _sanitize_task(self.task),
            "termination_reason": self.termination_reason.value if self.termination_reason is not None else None,
            "is_correct": bool(self.is_correct),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "metrics": self.metrics,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create Episode from dictionary, properly deserializing Trajectory objects."""
        from rllm.engine.agent_workflow_engine import TerminationReason

        return cls(
            id=data["id"],
            task=data["task"],
            termination_reason=TerminationReason(data["termination_reason"]) if data.get("termination_reason") is not None else TerminationReason.UNKNOWN,
            is_correct=data["is_correct"],
            trajectories=[Trajectory.from_dict(trajectory_data) for trajectory_data in data["trajectories"]],
            metrics=data.get("metrics", {}),
            info=data.get("info", {}),
        )


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
