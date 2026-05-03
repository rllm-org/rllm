"""Pydantic data models for the rllm-model-gateway."""

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


class TraceRecord(BaseModel):
    """A single captured LLM call with full token-level data."""

    trace_id: str
    session_id: str
    model: str = ""
    # rLLM session metadata — populated by middleware from headers/body/URL.
    # All optional so existing training writers (which only stamp session_id)
    # remain valid; eval and harness-stamped requests fill these in.
    run_id: str | None = None
    harness: str | None = None
    step_id: int | None = None
    parent_span_id: str | None = None
    span_type: str = "llm.call"
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


class UpstreamRoute(BaseModel):
    """Map a model name to an OpenAI-compatible upstream endpoint.

    The gateway treats this as opaque routing: it does not know about
    providers, auth shapes, or model-name conventions. The caller
    (typically rllm) resolves any provider knowledge — choice of
    upstream URL, the literal ``Authorization`` header value to use,
    and any params the upstream rejects — and hands the gateway a
    fully-resolved route to forward to.
    """

    upstream_url: str  # e.g. "http://127.0.0.1:4000/v1" or "https://api.openai.com/v1"
    # Full ``Authorization`` header value, e.g. ``"Bearer sk-..."``.
    # ``None`` means the gateway forwards no auth header upstream.
    auth_header: str | None = None
    # Body fields to strip before forwarding (e.g. ``"max_tokens"`` for
    # OpenAI o-series models that reject it). The gateway treats this as
    # a generic body-shaping list; it doesn't interpret which model needs
    # what.
    drop_params: list[str] = Field(default_factory=list)


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 9090
    workers: list[WorkerConfig] = Field(default_factory=list)
    # Map model name → upstream route. The gateway looks up
    # ``body["model"]`` and forwards to the matching route; falls
    # through to the worker-pool path if no match.
    routes: dict[str, UpstreamRoute] = Field(default_factory=dict)
    db_path: str | None = None
    store_worker: str = "sqlite"
    add_logprobs: bool = True
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = True
    routing_policy: str | None = None
    health_check_interval: float = 10.0
    log_level: str = "INFO"
    sync_traces: bool = False
    sampling_params_priority: str = "client"
    model: str | None = None  # When set, overrides ``body.model``
    # Identifier of the gateway run — eval CLI sets this to the run dir
    # basename, training/harness shims pick something globally unique.
    # When set, every trace persisted by this gateway is tagged with it
    # and a row is registered in the ``runs`` table on startup.
    run_id: str | None = None
    # Free-form metadata for the run (benchmark, model, agent, source,
    # …). Surfaced by the cross-run viewer.
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    # When set, every inbound request must carry
    # ``Authorization: Bearer <inbound_auth_token>``. Generated per-run
    # by the eval gateway when a public URL is exposed; never persisted.
    inbound_auth_token: str | None = None
