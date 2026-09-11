"""Pydantic data models for the rllm-model-gateway."""

import json
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, PrivateAttr, model_validator


class TraceRecord(BaseModel):
    """A single captured LLM call with full token-level data."""

    trace_id: str
    session_id: str
    # Which conversation lineage within the session this call belongs to
    # (SessionSlots assigns one per parent/subagent lineage). None when
    # cumulative-token mode is off (no slots) — the reader treats the whole
    # session as one lineage, as before.
    lineage_id: str | None = None
    model: str = ""
    # Input
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt_token_ids: list[int] = Field(default_factory=list)
    # Output
    response_message: dict[str, Any] = Field(default_factory=dict)
    completion_token_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] | None = None
    routing_matrices: list[str] | None = None
    finish_reason: str | None = None
    weight_version: int | None = None
    # Metadata
    latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


def _message_key(message: dict[str, Any]) -> str:
    """Exact compact-JSON identity for one chat-completion block."""
    key = json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if json.loads(key) != message:
        raise TypeError("chat-completion messages must contain only JSON-native values")
    return key


def canonicalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert equivalent chat-message wire shapes to one stable form.

    The gateway response uses ``reasoning`` while LiteLLM resends that same
    assistant turn as ``reasoning_content`` and duplicates it under
    ``provider_specific_fields.reasoning``. A null refusal is another
    transport-added default. Normalize only those known aliases; retain every
    other provider-specific field so materially different messages stay
    different.

    TraceRecord remains the untouched capture format. TraceDelta calls this at
    the raw-record-to-graph boundary, so the graph has one stable message shape.
    """
    normalized = json.loads(_message_key(message))
    provider_fields = normalized.get("provider_specific_fields")
    if provider_fields is None:
        normalized.pop("provider_specific_fields", None)
        provider_fields = None

    reasoning_values = [value for value in (normalized.get("reasoning"), normalized.get("reasoning_content")) if value is not None]
    if reasoning_values and all(value == reasoning_values[0] for value in reasoning_values[1:]):
        normalized.pop("reasoning", None)
        normalized.pop("reasoning_content", None)
        normalized["reasoning_content"] = reasoning_values[0]
        if isinstance(provider_fields, dict) and provider_fields.get("reasoning") == reasoning_values[0]:
            provider_fields.pop("reasoning", None)
    elif not reasoning_values:
        if normalized.get("reasoning") is None:
            normalized.pop("reasoning", None)
        if normalized.get("reasoning_content") is None:
            normalized.pop("reasoning_content", None)
        if isinstance(provider_fields, dict) and provider_fields.get("reasoning") is None:
            provider_fields.pop("reasoning", None)

    if normalized.get("refusal") is None:
        normalized.pop("refusal", None)
    if isinstance(provider_fields, dict):
        if provider_fields.get("refusal") is None:
            provider_fields.pop("refusal", None)
        if not provider_fields:
            normalized.pop("provider_specific_fields", None)
    return json.loads(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _canonical_message_key(message: dict[str, Any]) -> str:
    """Stable identity for parent discovery, independent of wire aliases."""
    return _message_key(canonicalize_message(message))


def _canonical_messages_start_with(values: list[dict[str, Any]], prefix: list[dict[str, Any]]) -> bool:
    return len(prefix) <= len(values) and all(_canonical_message_key(value) == _canonical_message_key(expected) for value, expected in zip(values[: len(prefix)], prefix, strict=True))


class TraceDelta(BaseModel):
    """One call stored against its completed conversational parent."""

    trace_id: str
    session_id: str
    parent_trace_id: str | None
    lineage_id: str | None = None
    model: str = ""
    messages_suffix: list[dict[str, Any]]
    prompt_ids_suffix: list[int]
    response_message: dict[str, Any]
    completion_token_ids: list[int]
    logprobs: list[float] | None = None
    routing_matrices: list[str] | None = None
    finish_reason: str | None = None
    weight_version: int | None = None
    latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _canonicalize_messages(self) -> "TraceDelta":
        self.messages_suffix = [canonicalize_message(message) for message in self.messages_suffix]
        self.response_message = canonicalize_message(self.response_message)
        return self

    @classmethod
    def against(
        cls,
        record: TraceRecord,
        parent: TraceRecord | None,
        *,
        _prefix_verified: bool = False,
    ) -> "TraceDelta":
        if record.raw_request is not None or record.raw_response is not None:
            raise ValueError(f"trace {record.trace_id!r}: raw_request/raw_response cannot be delta-stored; use the default store for raw capture")

        messages = [] if parent is None else [*parent.messages, parent.response_message]
        prompt_ids = [] if parent is None else [*parent.prompt_token_ids, *parent.completion_token_ids]

        if not (
            parent is not None
            and parent.session_id == record.session_id
            and parent.lineage_id == record.lineage_id
            and (_prefix_verified or _canonical_messages_start_with(record.messages, messages))
            and (_prefix_verified or record.prompt_token_ids[: len(prompt_ids)] == prompt_ids)
        ):
            parent = None
            messages = []
            prompt_ids = []
        return cls(
            **{name: getattr(record, name) for name in _SHARED_TRACE_FIELDS},
            parent_trace_id=None if parent is None else parent.trace_id,
            messages_suffix=record.messages[len(messages) :],
            prompt_ids_suffix=record.prompt_token_ids[len(prompt_ids) :],
        )


_SHARED_TRACE_FIELDS = TraceRecord.model_fields.keys() & TraceDelta.model_fields.keys()


class TraceGraph(BaseModel):
    """Append-only trace forest plus a private chat-message trie.

    A canonical renderer is assumed to produce one token tape for the same
    message path, lineage, model, and state. A mismatch safely becomes a root.
    """

    format: Literal["compact"]
    version: Literal[1]
    deltas: list[TraceDelta]

    _trace_positions: dict[str, int] = PrivateAttr(default_factory=dict)
    _message_children: dict[tuple[int, str], int] = PrivateAttr(default_factory=dict)
    _completed_nodes: dict[str, int] = PrivateAttr(default_factory=dict)
    _completed_representatives: dict[tuple[int, str | None, str], str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _rebuild_index(self) -> "TraceGraph":
        self._trace_positions = {}
        self._message_children = {}
        self._completed_nodes = {}
        self._completed_representatives = {}
        for delta in self.deltas:
            self._validate_append(delta)
            self._trace_positions[delta.trace_id] = len(self._trace_positions)
            self._index_delta_states(delta)
        return self

    def _validate_append(self, delta: TraceDelta) -> None:
        if delta.trace_id in self._trace_positions:
            raise ValueError(f"duplicate trace id {delta.trace_id!r}")
        if self.deltas and delta.session_id != self.deltas[0].session_id:
            raise ValueError(f"trace {delta.trace_id!r} belongs to another session")
        if delta.parent_trace_id is not None:
            parent = self.delta(delta.parent_trace_id)
            if parent is None:
                raise ValueError(f"trace {delta.trace_id!r}: parent is not earlier in the graph")
            if parent.session_id != delta.session_id or parent.lineage_id != delta.lineage_id:
                raise ValueError(f"trace {delta.trace_id!r}: parent belongs to another session or lineage")

    def _intern_message(self, node: int, message: dict[str, Any]) -> int:
        key = (node, _message_key(message))
        return self._message_children.setdefault(key, len(self._message_children) + 1)

    def _index_delta_states(self, delta: TraceDelta) -> None:
        node = 0 if delta.parent_trace_id is None else self._completed_nodes[delta.parent_trace_id]
        for message in delta.messages_suffix:
            node = self._intern_message(node, message)
        node = self._intern_message(node, delta.response_message)
        self._completed_nodes[delta.trace_id] = node
        self._completed_representatives.setdefault((node, delta.lineage_id, delta.model), delta.trace_id)

    def _find_parent(self, record: TraceRecord) -> TraceRecord | None:
        node = 0
        completed_trace_id = None
        for message in record.messages:
            next_node = self._message_children.get((node, _canonical_message_key(message)))
            if next_node is None:
                break
            node = next_node
            completed_trace_id = self._completed_representatives.get((node, record.lineage_id, record.model)) or completed_trace_id

        if completed_trace_id is None:
            return None
        parent = self.resolve(completed_trace_id)
        prompt_ids = [*parent.prompt_token_ids, *parent.completion_token_ids]
        return parent if record.prompt_token_ids[: len(prompt_ids)] == prompt_ids else None

    def add(self, record: TraceRecord) -> TraceDelta:
        parent = self._find_parent(record)
        delta = TraceDelta.against(record, parent, _prefix_verified=parent is not None)
        self.append(delta)
        return delta

    def replace_leaf(self, record: TraceRecord) -> TraceDelta:
        position = self._trace_positions.get(record.trace_id)
        if position is None:
            raise ValueError(f"unknown trace {record.trace_id!r}")
        if any(delta.parent_trace_id == record.trace_id for delta in self.deltas):
            raise ValueError(f"cannot replace trace {record.trace_id!r} after it has children")
        current = self.deltas[position]
        if record.session_id != current.session_id or record.lineage_id != current.lineage_id:
            raise ValueError(f"cannot move trace {record.trace_id!r} to another session or lineage")
        parent = None if current.parent_trace_id is None else self.resolve(current.parent_trace_id)
        replacement = TraceDelta.against(record, parent)
        self.deltas[position] = replacement
        self._rebuild_index()
        return replacement

    def append(self, delta: TraceDelta) -> None:
        self._validate_append(delta)
        for message in [*delta.messages_suffix, delta.response_message]:
            _message_key(message)
        self._trace_positions[delta.trace_id] = len(self.deltas)
        self.deltas.append(delta)
        self._index_delta_states(delta)

    def delta(self, trace_id: str) -> TraceDelta | None:
        at = self._trace_positions.get(trace_id)
        return None if at is None else self.deltas[at]

    def resolve(self, trace_id: str, memo: dict[str, TraceRecord] | None = None) -> TraceRecord:
        memo = {} if memo is None else memo
        if trace_id in memo:
            return memo[trace_id]
        chain: list[TraceDelta] = []
        seen: set[str] = set()
        tid = trace_id
        while tid not in memo:
            if tid in seen:
                raise ValueError(f"delta cycle at trace {tid!r}")
            seen.add(tid)
            delta = self.delta(tid)
            if delta is None:
                raise ValueError(f"unknown trace {tid!r}")
            chain.append(delta)
            if delta.parent_trace_id is None:
                break
            tid = delta.parent_trace_id
        leaf = chain[0]
        base = memo.get(tid)
        if base is None:
            root = chain.pop()
            messages = list(root.messages_suffix)
            prompt_ids = list(root.prompt_ids_suffix)
            previous: TraceRecord | TraceDelta = root
        else:
            messages = list(base.messages)
            prompt_ids = list(base.prompt_token_ids)
            previous = base
        for delta in reversed(chain):
            messages.append(previous.response_message)
            prompt_ids.extend(previous.completion_token_ids)
            messages.extend(delta.messages_suffix)
            prompt_ids.extend(delta.prompt_ids_suffix)
            previous = delta
        record = TraceRecord.model_construct(
            **{name: getattr(leaf, name) for name in _SHARED_TRACE_FIELDS},
            messages=messages,
            prompt_token_ids=prompt_ids,
            raw_request=None,
            raw_response=None,
        )
        memo[trace_id] = record
        return record

    def flatten(self) -> list[TraceRecord]:
        memo: dict[str, TraceRecord] = {}
        return [self.resolve(delta.trace_id, memo) for delta in self.deltas]

    def slice(self, trace_ids: list[str]) -> "TraceGraph":
        emitted: set[str] = set()
        out: list[TraceDelta] = []
        for tid in trace_ids:
            delta = self.delta(tid)
            if delta is None:
                continue
            if delta.parent_trace_id is not None and delta.parent_trace_id not in emitted:
                delta = TraceDelta.against(self.resolve(tid), None)
            out.append(delta)
            emitted.add(tid)
        return TraceGraph(format=self.format, version=self.version, deltas=out)


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
    store_worker: str = "memory"
    add_logprobs: bool = True
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = True
    routing_policy: str | None = None
    health_check_interval: float = 10.0
    log_level: str = "INFO"
    sync_traces: bool = False
    model: str | None = None  # When set, overrides ``body.model``
    cumulative_token_mode: bool = False
    # prime-rl family for the cumulative-mode bridge (e.g. "qwen3", "deepseek-v3"); "auto"
    # resolves from the model id. See rllm.renderers.get_renderer.
    renderer_family: str = "auto"
    # Tinker-style renderer name override for models prime-rl doesn't cover
    # (e.g. "deepseek_v4"), served via the Fireworks/Tinker cookbook renderer.
    renderer_name: str | None = None
