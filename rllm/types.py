"""Canonical lightweight types for rLLM.

These Pydantic BaseModel classes are the single source of truth for
Step, Trajectory, and Episode. The SDK uses them directly (via the
StepView / TrajectoryView aliases), while the training code in
``rllm.agents.agent`` extends them with training-specific fields
(token IDs, logprobs, advantage, etc.).
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    metadata: dict | None = None

    @property
    def result(self):
        """Get the output from the trajectory (backward compatibility)."""
        return self.output


class Episode(BaseModel):
    """A rollout episode containing one or more Trajectories."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = ""
    task: Any = None
    termination_reason: Any | None = None
    is_correct: bool = False
    trajectories: list[Trajectory] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.id.split(":")[0]

    @property
    def rollout_idx(self) -> str:
        return self.id.split(":")[1]


# Backward-compatible aliases for SDK code
StepView = Step
TrajectoryView = Trajectory
