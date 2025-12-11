"""Session management for RLLM SDK."""

from pathlib import Path
from typing import Any

from rllm.sdk.session.base import SessionProtocol, wrap_with_session_context
from rllm.sdk.session.contextvar import (
    ContextVarSession,
    get_active_cv_sessions,
    get_current_cv_metadata,
    get_current_cv_session,
    get_current_cv_session_name,
)
from rllm.sdk.session.session_buffer import (
    SessionBuffer,
    SessionBufferProtocol,
)


def _load_session_backend() -> str:
    """Load session backend configuration from config.yaml file.

    Returns:
        Session backend type: "contextvar" (default) or "opentelemetry"
    """
    # Try to read from rllm/sdk/config.yaml file
    config_file = Path(__file__).parent.parent / "config.yaml"

    if config_file.exists():
        try:
            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f)
                if config and "session_backend" in config:
                    value = config["session_backend"]
                    if value in ("contextvar", "opentelemetry"):
                        return value
        except Exception:
            pass  # Fall through to default

    # Default to contextvar
    return "contextvar"


# Global session backend configuration
SESSION_BACKEND = _load_session_backend()

# Conditionally import OTEL modules only if backend is "opentelemetry"
if SESSION_BACKEND == "opentelemetry":
    from rllm.sdk.session.opentelemetry import (
        OpenTelemetrySession,
        get_active_otel_session_uids,
        get_current_otel_metadata,
        get_current_otel_session_name,
        otel_session,
    )
else:
    # Stub definitions for type hints when OTEL not in use
    OpenTelemetrySession = None  # type: ignore
    get_active_otel_session_uids = None  # type: ignore
    get_current_otel_metadata = None  # type: ignore
    get_current_otel_session_name = None  # type: ignore
    otel_session = None  # type: ignore


# Public routing functions - these dispatch to the appropriate backend
def get_current_session() -> ContextVarSession | None:
    """Get the current session instance from context (ContextVar backend only).

    Note: OpenTelemetry backend uses baggage as source of truth and doesn't store
    session objects. Use get_current_metadata() or get_current_session_name() instead.
    """
    if SESSION_BACKEND == "opentelemetry":
        raise NotImplementedError("get_current_session() not supported with OpenTelemetry backend. Use get_current_metadata(), get_current_session_name(), or get_active_session_uids().")
    return get_current_cv_session()


def get_current_session_name() -> str | None:
    """Get current session name from context.

    Routes to the appropriate backend based on SESSION_BACKEND configuration.

    Returns:
        Session name or None if no session is active.
    """
    if SESSION_BACKEND == "opentelemetry":
        return get_current_otel_session_name()
    return get_current_cv_session_name()


def get_current_metadata() -> dict[str, Any]:
    """Get current metadata from context.

    Routes to the appropriate backend based on SESSION_BACKEND configuration.

    Returns:
        Metadata dict or empty dict if no metadata is set.
    """
    if SESSION_BACKEND == "opentelemetry":
        return get_current_otel_metadata()
    return get_current_cv_metadata()


def get_active_session_uids() -> list[str]:
    """Get active session UID list for the current backend.

    Returns:
        - contextvar backend: [outer_uid, ..., inner_uid]
        - opentelemetry backend: the active session's UID chain
        - no active session: []
    """
    if SESSION_BACKEND == "opentelemetry":
        return get_active_otel_session_uids()
    return [s._uid for s in get_active_cv_sessions()]


# Default session type (alias for convenience)
SessionContext = ContextVarSession

__all__ = [
    # Configuration
    "SESSION_BACKEND",
    # Protocol
    "SessionProtocol",
    # Implementations
    "ContextVarSession",
    "OpenTelemetrySession",
    "SessionContext",  # Default alias
    "otel_session",
    # Context helpers
    "get_current_session",
    "get_current_session_name",
    "get_current_metadata",
    "get_active_session_uids",
    # Runtime helpers
    "wrap_with_session_context",
    # Session buffer (ephemeral trace storage)
    "SessionBufferProtocol",  # Protocol
    "SessionBuffer",  # Default in-memory buffer
]
