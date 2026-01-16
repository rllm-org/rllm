"""Pydantic models for API request/response validation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# Session models
class SessionCreate(BaseModel):
    project: str
    experiment: str
    config: dict[str, Any] | None = None
    source_metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    id: str
    project: str
    experiment: str
    config: dict[str, Any] | None
    source_metadata: dict[str, Any] | None
    created_at: datetime
    completed_at: datetime | None = None


# Metrics models
class MetricsCreate(BaseModel):
    session_id: str
    step: int
    data: dict[str, Any]


class MetricsResponse(BaseModel):
    id: int
    session_id: str
    step: int
    data: dict[str, Any]
    created_at: datetime


# Episode models
class TrajectoryStep(BaseModel):
    observation: Any
    action: Any
    reward: float
    done: bool
    chat_completions: Any | None = None
    model_response: Any | None = None


class Trajectory(BaseModel):
    uid: str
    name: str | None = None
    reward: float
    steps: list[TrajectoryStep]


class EpisodeCreate(BaseModel):
    session_id: str
    step: int
    episode_id: str
    task: dict[str, Any]
    is_correct: bool
    reward: float | None = None
    trajectories: list[Trajectory]
    info: dict[str, Any] | None = None


class EpisodeResponse(BaseModel):
    id: str
    session_id: str
    step: int
    task: dict[str, Any]
    is_correct: bool
    reward: float | None
    data: dict[str, Any]  # Full episode data including trajectories
    created_at: datetime


# Health check
class HealthResponse(BaseModel):
    status: str
