"""Session management for RLLM SDK."""

from rllm.sdk.session.base import SessionProtocol
from rllm.sdk.session.contextvar import (
    ContextVarSession,
    get_active_sessions,
    get_current_metadata,
    get_current_session,
    get_current_session_name,
)
from rllm.sdk.session.storage import (
    InMemoryStorage,
    SessionStorage,
    SqliteSessionStorage,
)

# Default session type (alias for convenience)
SessionContext = ContextVarSession

__all__ = [
    # Protocol
    "SessionProtocol",
    # Implementations
    "ContextVarSession",
    "SessionContext",  # Default alias
    # Context helpers
    "get_current_session",
    "get_current_session_name",
    "get_current_metadata",
    "get_active_sessions",
    # Storage
    "SessionStorage",  # Protocol
    "InMemoryStorage",  # Default in-memory storage
    "SqliteSessionStorage",  # SQLite-backed storage
]
