"""Minimal Step/Trajectory types for standalone agent execution.

When running inside a sandbox with rllm[sdk] installed, agents can import
from ``rllm.types`` directly.  This module provides the same types for
environments where rllm is not available (e.g. AgentCore containers).
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
    input: dict | None = None
    output: Any = None
    metadata: dict | None = None
