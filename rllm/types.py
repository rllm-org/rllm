"""Canonical lightweight types for rLLM.

Single source of truth for the core data shapes (:class:`Task`,
:class:`Step`, :class:`Trajectory`, :class:`Episode`) and the
producer/consumer protocols around them (:class:`AgentFlow`,
:class:`Evaluator`, :class:`AgentConfig`, :func:`run_agent_flow`).

Eval-specific result shapes (:class:`~rllm.eval.types.EvalOutput`,
:class:`~rllm.eval.types.Signal`) live in :mod:`rllm.eval.types`.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from rllm.eval.types import EvalOutput


@dataclass
class Task:
    """A single problem instance.

    Pure data — describes itself (instruction, metadata) and points at the
    directory where its verifier lives. The :class:`rllm.runner.Runner`
    reads the task config and resolves the appropriate :class:`Evaluator`
    at run time.

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


class Step(BaseModel):
    """A single interaction step (one LLM call with optional reward)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Any | None = None
    output: Any | None = None
    action: Any | None = None
    reward: float = 0.0
    done: bool = False
    metadata: dict | None = None


class Trajectory(BaseModel):
    """A sequence of Steps forming one agent trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "agent"
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


@runtime_checkable
class AgentFlow(Protocol):
    """A runnable agent program that produces an :class:`Episode`.

    An AgentFlow may orchestrate one or many agents internally; each
    contributes one or more Trajectories to the resulting Episode.

    Implementations may provide either ``run`` (sync) or ``arun``
    (async). If both are present, callers should prefer ``arun`` when
    running inside an event loop — see :func:`run_agent_flow`.
    """

    def run(self, task: Any, config: AgentConfig) -> Episode: ...


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
    """
    if hasattr(agent, "arun") and inspect.iscoroutinefunction(agent.arun):
        return await agent.arun(task, config)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, agent.run, task, config)
