"""RLLM SDK for automatic LLM trace collection and RL training."""

from rllm.sdk.decorators import trajectory
from rllm.sdk.protocol import StepView, Trace, TrajectoryView
from rllm.sdk.session import (
    ContextVarSession,
    SessionBuffer,
    SessionBufferProtocol,
    SessionContext,
    get_current_metadata,
    get_current_session,
    get_current_session_name,
)
from rllm.sdk.shortcuts import get_chat_client, get_chat_client_async, session
from rllm.sdk.tracers import (
    InMemorySessionTracer,
    SqliteTracer,
    TracerProtocol,
)

__all__ = [
    # Protocol / Data Models
    "Trace",  # Low-level LLM call trace
    "StepView",  # Trace wrapper with reward field (auto-generated from traces)
    "TrajectoryView",  # Collection of steps forming a workflow
    # Decorators
    "trajectory",  # Decorator to mark function as trajectory (returns TrajectoryView)
    # Sessions
    "SessionContext",  # Default (alias for ContextVarSession)
    "ContextVarSession",  # Explicit contextvars-based session
    "get_current_session",  # Get current session instance
    "get_current_session_name",  # Get current session name
    "get_current_metadata",  # Get current metadata
    # Session Buffer (ephemeral trace storage)
    "SessionBufferProtocol",  # Buffer protocol
    "SessionBuffer",  # Default in-memory buffer
    # Shortcuts
    "session",
    "get_chat_client",
    "get_chat_client_async",
    # Tracers
    "TracerProtocol",  # Tracer interface
    "InMemorySessionTracer",  # In-memory tracer for immediate access
    "SqliteTracer",  # SQLite-based persistent tracer
]
