"""Canonical lightweight types for rLLM.

Single source of truth for the core data shapes (:class:`Task`,
:class:`Step`, :class:`Trajectory`, :class:`Episode`, :class:`Action`,
:class:`TrajectoryGroup`) and the producer/consumer protocols around
them (:class:`AgentFlow`, :class:`Evaluator`, :class:`AgentConfig`,
:func:`run_agent_flow`).

Eval-specific result shapes (:class:`~rllm.eval.types.EvalOutput`,
:class:`~rllm.eval.types.Signal`) live in :mod:`rllm.eval.types`.

Historically the training-side (token IDs, logprobs, advantages, ...)
fields lived on subclasses in ``rllm.agents.agent``. They have been
folded into the canonical classes here; ``rllm.agents.agent`` remains
as a backward-compat re-export shim.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rllm.engine.rollout import ModelOutput
    from rllm.eval.types import EvalOutput


@dataclass
class Task:
    """A single problem instance.

    Pure data — describes itself (instruction, metadata) and points at the
    directory where its verifier lives.
    :class:`rllm.experimental.engine.agent_flow_engine.AgentFlowEngine`
    (driven by :class:`rllm.eval._hooks.EvalHooks` at eval time) reads
    the task config and resolves the appropriate :class:`Evaluator` at
    run time.

    Two physical shapes both produce ``Task`` instances:

    1. **Task-per-directory** (Harbor-style): each ``task-NNN/`` is one
       Task with ``sub_dir`` set to its subdirectory. Verifier lives in
       ``dataset_dir/sub_dir/tests/``.

    2. **Rows-with-shared-verifier** (gsm8k-style): one row from a JSONL
       file becomes one Task. ``sub_dir`` is ``None``; the verifier is
       shared across all rows and lives in ``dataset_dir/tests/`` (or
       is referenced by name in ``dataset.toml``).
    """

    id: str
    """Stable identifier (e.g. row index, task-NNN, or harbor task name)."""

    instruction: str | list[dict]
    """What the agent sees. Plain text, or multimodal content blocks."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary task data (ground truth, MCQ choices, harbor task.toml, ...).
    For data tasks, this is the source row. For sandbox tasks, this is
    the parsed ``task.toml`` plus anything else the verifier needs."""

    dataset_dir: Path = field(default_factory=Path)
    """Path to the dataset directory (where ``dataset.toml`` lives)."""

    sub_dir: Path | None = None
    """For task-per-directory shape: relative path of this task's subdir
    within ``dataset_dir``. ``None`` for rows-with-shared-verifier."""

    @property
    def task_dir(self) -> Path:
        """The directory holding *this* task's files.

        For per-task-dir tasks: ``dataset_dir / sub_dir``.
        For shared-verifier tasks: ``dataset_dir`` (verifier is shared).
        """
        return self.dataset_dir / self.sub_dir if self.sub_dir else self.dataset_dir


# ---------------------------------------------------------------------------
# Trajectory data types (formerly defined in rllm.agents.agent)
# ---------------------------------------------------------------------------

_DEFAULT_TRAJ_NAME = "default_traj_name"


class Action(BaseModel):
    """Wraps an arbitrary action emitted by an agent."""

    action: Any = None


class Step(BaseModel):
    """A single interaction step (one LLM call with optional reward).

    Fields are split into two groups:

    - **Core / eval**: ``id``, ``input``, ``output``, ``action``,
      ``reward``, ``done``, ``metadata`` — populated by every code path
      (harness, eval, training).
    - **Training payloads**: ``prompt_ids``, ``response_ids``, ``logprobs``,
      ``chat_completions``, ``model_output``, ``advantage`` etc. —
      populated by training rollouts; default-empty in eval-only paths.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    # Core / eval fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Any | None = None
    output: Any | None = None
    action: Any | None = None
    reward: float = 0.0
    done: bool = False
    metadata: dict | None = None

    # Training-side payloads
    prompt_ids: list[int] | list[Any] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] = Field(default_factory=list)
    routing_matrices: list[str] | None = None  # per-token routing matrices (R3, transient)
    chat_completions: list[dict[str, Any]] = Field(default_factory=list)
    observation: Any = None
    thought: str = ""
    model_response: str = ""
    model_output: Any = None  # Runtime type is ModelOutput | None; uses Any to avoid circular import
    mc_return: float = 0.0
    advantage: list[float] | float | None = None
    weight_version: int | None = None  # weight version at time of generation (async staleness)

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
        self.chat_completions = deepcopy(self.chat_completions)
        if self.model_output is None:
            return
        # backfill fields like prompt_ids, response_ids, logprobs, etc.
        if len(self.prompt_ids) == 0 and self.model_output.prompt_ids is not None:
            self.prompt_ids = self.model_output.prompt_ids
        if len(self.response_ids) == 0 and self.model_output.completion_ids is not None:
            self.response_ids = self.model_output.completion_ids
        if len(self.logprobs) == 0 and self.model_output.logprobs is not None:
            self.logprobs = self.model_output.logprobs
        if self.routing_matrices is None and getattr(self.model_output, "routing_matrices", None) is not None:
            self.routing_matrices = self.model_output.routing_matrices
        if self.weight_version is None and hasattr(self.model_output, "weight_version"):
            self.weight_version = self.model_output.weight_version

        # check that the lengths would match up
        if len(self.logprobs) > 0:
            assert len(self.response_ids) == len(self.logprobs), f"length mismatch between response_ids and logprobs, got {len(self.response_ids)}, {len(self.logprobs)}"

    def to_dict(self) -> dict:
        from rllm.tools.tool_base import ToolCall, ToolOutput

        # Helper function to recursively convert ToolCall and ToolOutput objects to dicts
        def _serialize_value(value):
            if isinstance(value, ToolCall | ToolOutput):
                return value.to_dict()
            elif isinstance(value, list):
                return [_serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            else:
                return value

        return {
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "logprobs": self.logprobs,
            "routing_matrices": self.routing_matrices,
            "chat_completions": _serialize_value(self.chat_completions),
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
            "weight_version": self.weight_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        from rllm.engine.rollout import ModelOutput

        return cls(
            prompt_ids=data["prompt_ids"],
            response_ids=data["response_ids"],
            logprobs=data["logprobs"],
            routing_matrices=data.get("routing_matrices"),
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
            advantage=data.get("advantage", 0.0),
            weight_version=data.get("weight_version"),
        )

    @classmethod
    def from_model_output(cls, model_output: ModelOutput, messages: list[dict] | None = None, action: Any | None = None) -> Step:
        return cls(
            prompt_ids=model_output.prompt_ids or [],
            response_ids=model_output.completion_ids or [],
            logprobs=model_output.logprobs or [],
            routing_matrices=getattr(model_output, "routing_matrices", None),
            chat_completions=(messages or []) + [{"role": "assistant", "content": model_output.content, "reasoning": model_output.reasoning}],
            thought=model_output.reasoning or "",
            action=action,
            model_response=model_output.content or "",
            model_output=model_output,
            weight_version=model_output.weight_version,
        )


class Trajectory(BaseModel):
    """A sequence of Steps forming one agent trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = _DEFAULT_TRAJ_NAME
    task: Any = None
    steps: list[Step] = Field(default_factory=list)
    reward: float | None = None
    input: dict | None = None  # Function arguments (SDK usage)
    output: Any = None  # Function return value (SDK usage)
    signals: dict[str, float] = Field(default_factory=dict)  # Evaluation signals
    metadata: dict | None = None

    @property
    def result(self):
        """Get the output from the trajectory (backward compatibility)."""
        return self.output

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


class Episode(BaseModel):
    """A rollout episode containing one or more Trajectories."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Any = None
    termination_reason: Any | None = None
    is_correct: bool = False
    trajectories: list[Trajectory] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.id.split(":")[0]

    @property
    def rollout_idx(self) -> str:
        return self.id.split(":")[1]

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
    weight_version: int = 0

    @property
    def group_role(self) -> str:
        return self.group_id.split(":")[1] if ":" in self.group_id[:-1] else "all_groups"

    @property
    def task_id(self) -> str:
        return self.group_id.split(":")[0]


# ---------------------------------------------------------------------------
# Core protocols + agent config
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration injected into every :class:`AgentFlow` call."""

    base_url: str
    model: str
    session_uid: str
    metadata: dict = field(default_factory=dict)
    is_validation: bool = False


@runtime_checkable
class AgentFlow(Protocol):
    """A runnable agent program that produces an :class:`Episode`.

    An AgentFlow may orchestrate one or many agents internally; each
    contributes one or more Trajectories to the resulting Episode.

    Implementations may provide either ``run`` (sync) or ``arun``
    (async). If both are present, callers should prefer ``arun`` when
    running inside an event loop — see :func:`run_agent_flow`.

    Return value: ``Episode`` for full control (multi-trajectory flows
    must use this), a ``Trajectory`` (auto-wrapped in an Episode), or
    ``None`` (framework builds an Episode with one empty Trajectory;
    gateway traces fill in the Steps during enrichment). The evaluator
    is responsible for parsing whatever it needs out of the resulting
    trajectories.
    """

    def run(self, task: Any, config: AgentConfig) -> Episode | Trajectory | None: ...


def _coerce_to_episode(result: Any, task: Any, traj_name: str) -> Episode:
    """Normalize an ``AgentFlow`` return value into an :class:`Episode`.

    Accepts:

    * :class:`Episode` — passed through (``task`` filled if missing).
    * :class:`Trajectory` — wrapped in ``Episode(trajectories=[t])``.
      The trajectory is left untouched; the evaluator is responsible
      for parsing whatever the user put on it.
    * ``None`` — framework builds an empty single-trajectory Episode.
      Gateway traces populate the Steps during enrichment; the
      evaluator reads what it needs out of those steps.

    Anything else raises :class:`TypeError`.
    """
    task_metadata = getattr(task, "metadata", task)

    if isinstance(result, Episode):
        if result.task is None:
            result.task = task_metadata
        return result

    if isinstance(result, Trajectory):
        if result.name == _DEFAULT_TRAJ_NAME:
            result.name = traj_name
        return Episode(task=task_metadata, trajectories=[result])

    if result is None:
        traj = Trajectory(name=traj_name, steps=[])
        return Episode(task=task_metadata, trajectories=[traj])

    raise TypeError(f"AgentFlow returned unsupported type {type(result).__name__}; expected Episode, Trajectory, or None")


@runtime_checkable
class Evaluator(Protocol):
    """Scores an :class:`Episode` produced by an :class:`AgentFlow`.

    The evaluator examines the task + episode trajectories and returns
    an :class:`~rllm.eval.output.EvalOutput`. The runner then writes the
    reward back onto each Trajectory, making them ready for RL training.
    """

    def evaluate(self, task: Any, episode: Episode) -> EvalOutput: ...


async def run_agent_flow(
    agent: AgentFlow,
    task: Any,
    config: AgentConfig,
    executor=None,
) -> Episode:
    """Run an :class:`AgentFlow`, preferring its async ``arun`` when present.

    Falls back to running ``run`` in *executor* (a ``ThreadPoolExecutor``)
    so that sync agent flows don't block the event loop.

    The return value is coerced into an :class:`Episode` via
    :func:`_coerce_to_episode`, so flows may return ``Episode``,
    ``Trajectory``, or ``None``.
    """
    if hasattr(agent, "arun") and inspect.iscoroutinefunction(agent.arun):
        result = await agent.arun(task, config)
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, agent.run, task, config)

    traj_name = getattr(agent, "name", None) or _DEFAULT_TRAJ_NAME
    return _coerce_to_episode(result, task, traj_name)
