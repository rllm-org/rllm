"""Pydantic data models for the rllm-model-gateway."""

from typing import Any

from pydantic import BaseModel, Field


class TraceRecord(BaseModel):
    """A single captured LLM call with full token-level data."""

    trace_id: str
    session_id: str
    model: str = ""
    # Input
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt_token_ids: list[int] = Field(default_factory=list)
    # Output
    response_message: dict[str, Any] = Field(default_factory=dict)
    completion_token_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] | None = None
    finish_reason: str | None = None
    # Metadata
    latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


class WorkerConfig(BaseModel):
    """Configuration for a single inference worker."""

    worker_id: str = ""
    url: str
    model_name: str | None = None
    weight: int = 1


class WorkerInfo(BaseModel):
    """Runtime info for a worker including health state."""

    worker_id: str
    url: str
    model_name: str | None = None
    weight: int = 1
    healthy: bool = True
    active_requests: int = 0


class SessionInfo(BaseModel):
    """Session metadata returned by session management APIs."""

    session_id: str
    trace_count: int = 0
    created_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 9090
    workers: list[WorkerConfig] = Field(default_factory=list)
    db_path: str | None = None
    store_worker: str = "sqlite"
    add_logprobs: bool = True
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = True
    routing_policy: str | None = None
    health_check_interval: float = 10.0
    log_level: str = "INFO"
    sync_traces: bool = False
