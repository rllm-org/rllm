"""Pydantic models for Rllm agent trace data.

Hierarchy (mirrors the ADK execution model):

    Session            — long-lived conversation between a user and an app
    └─ Invocation      — one user message → final response cycle
         ├─ AgentSpan  — one agent.run_async() call
         │    ├─ LlmSpan   — one LLM generate_content call
         │    └─ ToolSpan  — one tool execution
         └─ EventRecord — each Event yielded by the runner

Each record type maps 1:1 to an ADK plugin callback pair:

    before_run / after_run       → Invocation
    before_agent / after_agent   → AgentSpan
    before_model / after_model   → LlmSpan
    before_tool / after_tool     → ToolSpan
    on_event                     → EventRecord
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------


class AgentInfo(BaseModel):
    """Metadata about an agent in the agent tree."""

    name: str
    """The agent name (e.g. ``"planner"``)."""

    description: str | None = None
    """Human-readable description of the agent."""

    type: str
    """Python class name — ``"LlmAgent"``, ``"SequentialAgent"``, etc."""

    sub_agents: list[AgentInfo] = Field(default_factory=list)
    """Recursive children in the agent hierarchy."""


class ToolInfo(BaseModel):
    """Lightweight tool descriptor extracted from the LLM request."""

    name: str
    description: str | None = None


class GenerationConfig(BaseModel):
    """Subset of ``GenerateContentConfig`` relevant for observability."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    candidate_count: int | None = None


class UsageMetadata(BaseModel):
    """Token usage counters from the LLM response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_content_token_count: int | None = None


class LlmRequest(BaseModel):
    """Captured LLM request payload.

    Populated from ``google.adk.models.LlmRequest`` in
    ``before_model_callback``.
    """

    model: str | None = None
    """Model name (e.g. ``"gemini-2.0-flash"``)."""

    system_instruction: str | None = None
    """The system prompt sent to the model."""

    contents: list[dict[str, Any]] | None = None
    """Serialised chat history (``list[types.Content]``)."""

    tools: list[ToolInfo] | None = None
    """Tool schemas available to the model for this request."""

    generation_config: GenerationConfig | None = None


class LlmResponseData(BaseModel):
    """Captured LLM response payload.

    Populated from ``google.adk.models.LlmResponse`` in
    ``after_model_callback``.
    """

    content: dict[str, Any] | None = None
    """Serialised ``types.Content`` from the model."""

    finish_reason: str | None = None
    model_version: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    usage: UsageMetadata | None = None


class EventActionsData(BaseModel):
    """Subset of ``EventActions`` relevant for observability."""

    state_delta: dict[str, Any] | None = None
    artifact_delta: dict[str, int] | None = None
    transfer_to_agent: str | None = None
    escalate: bool | None = None
    end_of_agent: bool | None = None


# ---------------------------------------------------------------------------
# Top-level trace records
# ---------------------------------------------------------------------------

# Discriminator values used in the NDJSON wire format ``{"type": ...}``.
SpanType = Literal[
    "session",
    "session.start",
    "invocation.start",
    "invocation.end",
    "agent.start",
    "agent.end",
    "llm.start",
    "llm.end",
    "tool.start",
    "tool.end",
    "tool.data",
    "event",
    "experiment.start",
    "experiment.end",
    "experiment.case",
    "episode.start",
    "episode.end",
    "trajectory.start",
    "trajectory.step",
    "trajectory.end",
]


class SessionRecord(BaseModel):
    """Lazily created on the first invocation for a given ``session_id``.

    There is **no** separate "create session" API — the plugin registers
    the session automatically.
    """

    session_id: str
    app_name: str
    user_id: str
    created_at: float


class SessionStartRecord(BaseModel):
    """Captures the initial state of a session — system prompt, user input,
    and any other context available at session start.

    Emitted once per session before the first invocation begins.
    """

    session_id: str
    agent_name: str | None = None
    started_at: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    """Free-form payload: system prompt, user message, case metadata, etc."""


class InvocationRecord(BaseModel):
    """One user-message → final-response cycle.

    Opened by ``before_run_callback``, closed by ``after_run_callback``.
    """

    invocation_id: str
    session_id: str
    app_name: str
    user_id: str

    # Input (captured in on_user_message_callback)
    user_message: dict[str, Any] | None = None
    """Serialised ``types.Content`` of the user's input."""

    # Agent tree snapshot (captured once at invocation start)
    root_agent: AgentInfo | None = None

    # Timing
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None

    # Aggregated metrics (populated at after_run)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    error_count: int = 0


class AgentSpanRecord(BaseModel):
    """One ``agent.run_async()`` execution.

    Opened by ``before_agent_callback``, closed by ``after_agent_callback``.
    """

    span_id: str
    invocation_id: str
    session_id: str

    agent_name: str
    agent_description: str | None = None
    agent_type: str
    """Python class name of the agent (e.g. ``"LlmAgent"``)."""

    branch: str | None = None
    """Dot-separated agent hierarchy path (e.g. ``"root.planner"``)."""

    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None

    output: dict[str, Any] | None = None
    """Serialised ``types.Content`` if the after-callback produced output."""

    error: str | None = None


class LlmSpanRecord(BaseModel):
    """One LLM ``generate_content`` call.

    Opened by ``before_model_callback``, closed by ``after_model_callback``.
    """

    span_id: str
    invocation_id: str
    session_id: str
    agent_name: str

    request: LlmRequest | None = None
    response: LlmResponseData | None = None

    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None


class ToolSpanRecord(BaseModel):
    """One tool execution.

    Opened by ``before_tool_callback``, closed by ``after_tool_callback``.
    """

    span_id: str
    invocation_id: str
    session_id: str
    agent_name: str

    tool_name: str
    tool_description: str | None = None
    tool_type: str
    """Python class name (e.g. ``"FunctionTool"``, ``"McpTool"``)."""

    is_long_running: bool = False

    args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None


class ToolDataRecord(BaseModel):
    """Raw data returned by a tool execution.

    Captures the observation/output payload from a tool call, separate
    from the tool.start/tool.end lifecycle spans.  Useful for recording
    large tool outputs (e.g. API responses, query results) that inform
    the agent's next reasoning step.
    """

    span_id: str
    session_id: str
    agent_name: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    """Free-form payload: observation data, summary, display name, etc."""


class EventRecord(BaseModel):
    """Raw ADK ``Event`` passthrough.

    Created for every event yielded by the runner (``on_event_callback``).
    """

    event_id: str
    invocation_id: str
    session_id: str

    author: str
    """``"user"`` or the agent name."""

    timestamp: float
    branch: str | None = None

    content: dict[str, Any] | None = None
    """Serialised ``types.Content``."""

    partial: bool | None = None
    is_final_response: bool = False

    actions: EventActionsData | None = None

    error_code: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Experiment records
# ---------------------------------------------------------------------------


class ScoreRecord(BaseModel):
    """Result of a single scorer on one test case."""

    name: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentCaseRecord(BaseModel):
    """One test case within an experiment run."""

    case_id: str
    experiment_id: str
    input: Any
    expected: Any | None = None
    output: Any | None = None
    scores: list[ScoreRecord] = Field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None


class ExperimentSummary(BaseModel):
    """Aggregated scores across all cases in an experiment."""

    scores: dict[str, float] = Field(default_factory=dict)
    """Scorer name → mean score across all cases."""

    total_cases: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0


class ExperimentRecord(BaseModel):
    """Metadata for a single experiment run."""

    experiment_id: str
    name: str
    created_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    summary: ExperimentSummary | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wire envelope — used in NDJSON stream
# ---------------------------------------------------------------------------


class TraceEnvelope(BaseModel):
    """Wrapper sent over the wire.  One JSON line per record.

    Example::

        {"type": "llm.start", "data": { ... }}
    """

    type: SpanType
    data: dict[str, Any]
