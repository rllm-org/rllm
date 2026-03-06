"""Pydantic data models for the rllm-model-gateway."""

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


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


def _split_worker_url(raw: str) -> dict[str, str]:
    """Split ``http://host:port/v1`` into base URL + api_path.

    If the URL contains a path component (e.g. ``/v1``), it is separated
    out so that health checks can use the bare ``scheme://host:port`` while
    proxying uses ``scheme://host:port + api_path``.
    """
    parsed = urlparse(raw.rstrip("/"))
    if parsed.path and parsed.path != "/":
        base = f"{parsed.scheme}://{parsed.netloc}"
        return {"url": base, "api_path": parsed.path}
    return {"url": raw.rstrip("/"), "api_path": "/v1"}


class WorkerConfig(BaseModel):
    """Configuration for a single inference worker."""

    worker_id: str = ""
    url: str  # base URL, e.g. "http://localhost:4000"
    api_path: str = "/v1"  # API version prefix, appended for proxying
    model_name: str | None = None
    weight: int = 1

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Backward compat: auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            # Only auto-split if api_path was NOT explicitly provided
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values


class WorkerInfo(BaseModel):
    """Runtime info for a worker including health state."""

    worker_id: str
    url: str  # base URL
    api_path: str = "/v1"
    model_name: str | None = None
    weight: int = 1
    healthy: bool = True
    active_requests: int = 0

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values

    @property
    def api_url(self) -> str:
        """Full URL for API proxying: base + api_path."""
        return self.url.rstrip("/") + self.api_path


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
