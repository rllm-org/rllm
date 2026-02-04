from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rllm.engine.rollout import ModelOutput
    from rllm.workflows.workflow import TerminationReason


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
    model_output: ModelOutput | None = None
    info: dict = field(default_factory=dict)  # Store any additional info.

    # field below are filled by the engine
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

    # field below are filled by the advantage computer. Note when advantage is a list, it is per-token advantages.
    # TODO: potentially rename this as "advantages" so its clearer that it allows a generic list.
    advantage: list[float] | float | None = None

    def __post_init__(self):
        if self.model_output is None:
            return
        # backfill fields like prompt_ids, response_ids, logprobs, etc.
        if len(self.prompt_ids) == 0 and self.model_output.prompt_ids is not None:
            self.prompt_ids = self.model_output.prompt_ids
        if len(self.response_ids) == 0 and self.model_output.completion_ids is not None:
            self.response_ids = self.model_output.completion_ids
        if len(self.logprobs) == 0 and self.model_output.logprobs is not None:
            self.logprobs = self.model_output.logprobs

        # check that the token ids are filled
        # TODO(listar2000): this might cause compatibility issue. Double check if we should make these assertions.
        # assert len(self.prompt_ids) > 0, "prompt_ids is empty"
        # assert len(self.response_ids) > 0, "response_ids is empty"

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
            info=data.get("info", {}),
            reward=data["reward"],
            done=data["done"],
            mc_return=data["mc_return"],
            advantage=data["advantage"],
        )


@dataclass
class Action:
    action: Any = None


_DEFAULT_TRAJ_NAME = "default_traj_name"


@dataclass
class Trajectory:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))  # unique id to deduplicate on
    name: str = _DEFAULT_TRAJ_NAME
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float | None = None  # it is possible that the trajectory-level reward does not exist
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
    termination_reason: TerminationReason | None = None  # noqa: F821
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
    def from_dict(cls, data: dict) -> Episode:
        """Create Episode from dictionary, properly deserializing Trajectory objects."""
        from rllm.engine.agent_workflow_engine import TerminationReason

        return cls(
            id=data["id"],
            task=data["task"],
            termination_reason=TerminationReason(data.get("termination_reason", TerminationReason.UNKNOWN)),
            is_correct=data["is_correct"],
            trajectories=[Trajectory.from_dict(trajectory_data) for trajectory_data in data["trajectories"]],
            metrics=data.get("metrics", {}),
            info=data.get("info", {}),
        )


@dataclass
class TrajectoryGroup:
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
    metadata: list[dict] = field(default_factory=list)

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
