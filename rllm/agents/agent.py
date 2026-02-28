from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from rllm.types import Episode as _EpisodeBase
from rllm.types import Step as _StepBase
from rllm.types import Trajectory as _TrajectoryBase

if TYPE_CHECKING:
    from rllm.engine.rollout import ModelOutput


class Step(_StepBase):
    """Training step with token IDs, logprobs, advantage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    # Training-specific fields
    prompt_ids: list[int] | list[Any] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)

    chat_completions: list[dict[str, str]] = Field(default_factory=list)

    observation: Any = None
    thought: str = ""
    # action: inherited from _StepBase
    model_response: str = ""
    model_output: Any = None  # Runtime type is ModelOutput | None; uses Any to avoid circular import

    # reward, done: inherited from _StepBase
    mc_return: float = 0.0

    # Per-token or scalar advantages
    advantage: list[float] | float | None = None

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None so mutation works."""
        if self.metadata is None:
            self.metadata = {}
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

    def model_post_init(self, __context: Any) -> None:
        if self.model_output is None:
            return
        # backfill fields like prompt_ids, response_ids, logprobs, etc.
        if len(self.prompt_ids) == 0 and self.model_output.prompt_ids is not None:
            self.prompt_ids = self.model_output.prompt_ids
        if len(self.response_ids) == 0 and self.model_output.completion_ids is not None:
            self.response_ids = self.model_output.completion_ids
        if len(self.logprobs) == 0 and self.model_output.logprobs is not None:
            self.logprobs = self.model_output.logprobs

        # check that the lengths would match up
        if len(self.logprobs) > 0:
            assert len(self.response_ids) == len(self.logprobs), f"length mismatch between response_ids and logprobs, got {len(self.response_ids)}, {len(self.logprobs)}"

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
            "advantage": self.advantage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
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
            metadata=data.get("info", data.get("metadata", {})),
            reward=data["reward"],
            done=data["done"],
            mc_return=data["mc_return"],
            advantage=data["advantage"],
        )

    @classmethod
    def from_model_output(cls, model_output: ModelOutput, messages: list[dict] | None = None, action: Any | None = None) -> Step:
        return cls(
            prompt_ids=model_output.prompt_ids or [],
            response_ids=model_output.completion_ids or [],
            logprobs=model_output.logprobs or [],
            chat_completions=(messages or []) + [{"role": "assistant", "content": model_output.content, "reasoning": model_output.reasoning}],
            thought=model_output.reasoning or "",
            action=action,
            model_response=model_output.content or "",
            model_output=model_output,
        )


class Action(BaseModel):
    action: Any = None


_DEFAULT_TRAJ_NAME = "default_traj_name"


class Trajectory(_TrajectoryBase):
    """Training trajectory extending the canonical Trajectory with core defaults."""

    name: str = _DEFAULT_TRAJ_NAME  # Override canonical default for core compat
    steps: list[Step] = Field(default_factory=list)  # Narrow type to training Step

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None so mutation works."""
        if self.metadata is None:
            self.metadata = {}
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

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
            "reward": float(self.reward) if self.reward is not None else None,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Trajectory:
        """Create Trajectory from dictionary, properly deserializing Step objects."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data["name"],
            task=data["task"],
            steps=[Step.from_dict(step_data) for step_data in data.get("steps", [])],
            reward=data["reward"],
            metadata=data.get("info", data.get("metadata", {})),
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


class Episode(_EpisodeBase):
    """Training episode extending the canonical Episode."""

    trajectories: list[Trajectory] = Field(default_factory=list)  # Narrow type

    @property
    def info(self) -> dict:
        """Alias for metadata. Auto-initializes to {} if None."""
        return self.metadata

    @info.setter
    def info(self, value: dict) -> None:
        self.metadata = value

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
    def from_dict(cls, data: dict) -> Episode:
        """Create Episode from dictionary, properly deserializing Trajectory objects."""
        from rllm.workflows.workflow import TerminationReason

        return cls(
            id=data["id"],
            task=data["task"],
            termination_reason=TerminationReason(data.get("termination_reason", TerminationReason.UNKNOWN)),
            is_correct=data["is_correct"],
            trajectories=[Trajectory.from_dict(trajectory_data) for trajectory_data in data["trajectories"]],
            metrics=data.get("metrics", {}),
            metadata=data.get("info", data.get("metadata", {})),
        )

    @property
    def task_id(self) -> str:
        return self.id.split(":")[0]

    @property
    def rollout_idx(self) -> str:
        return self.id.split(":")[1]


class TrajectoryGroup(BaseModel):
    """
    A group of trajectories for advantage computation.

    Unlike Episode (which represents raw rollout data), TrajectoryGroup is specifically
    structured for advantage computation. All trajectories in a group will have their
    rewards compared to compute advantages (e.g., via GRPO).

    Attributes:
        trajectories: List of trajectories to compare for advantage computation
        group_id: Optional identifier for the group (e.g., "task1:agent_0")
        metadata: List of metadata for each trajectory in the group
    """

    trajectories: list[Trajectory]
    group_id: str = ""
    metadata: list[dict] = Field(default_factory=list)

    @property
    def group_role(self) -> str:
        return self.group_id.split(":")[1] if ":" in self.group_id[:-1] else "all_groups"

    @property
    def task_id(self) -> str:
        return self.group_id.split(":")[0]


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
