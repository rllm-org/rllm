"""ContextVar-based session implementation (default)."""

import contextvars
import uuid
from collections.abc import Callable
from typing import Any

from rllm.sdk.protocol import StepView, Trace, trace_to_step_view
from rllm.sdk.session.session_buffer import SessionBuffer

# Session-specific context variables
_current_session: contextvars.ContextVar["ContextVarSession | None"] = contextvars.ContextVar("current_session", default=None)
_session_name: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_name", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
# Stack of active sessions (outer → inner). Use None default to avoid shared list instances.
_sessions_stack: contextvars.ContextVar[list["ContextVarSession"] | None] = contextvars.ContextVar("sessions_stack", default=None)


def get_current_cv_session() -> "ContextVarSession | None":
    """Get the current ContextVarSession from contextvars.

    Returns:
        The active ContextVarSession or None if no session is active.
    """
    return _current_session.get()


def get_current_cv_session_name() -> str | None:
    """Get current session name from ContextVar.

    Returns:
        Session name or None if no session is active.
    """
    return _session_name.get()


def get_current_cv_metadata() -> dict[str, Any]:
    """Get current metadata from ContextVar.

    Returns:
        Metadata dict or empty dict if no metadata is set.
    """
    metadata = _metadata.get()
    return metadata if metadata is not None else {}


def get_active_cv_sessions() -> list["ContextVarSession"]:
    """Get a copy of the current stack of active ContextVarSessions (outer → inner).

    Returns:
        List of active ContextVarSessions or empty list.
    """
    stack = _sessions_stack.get() or []
    # Return a shallow copy to prevent accidental mutation by callers
    return list(stack)


class ContextVarSession:
    """Context-based session for in-process LLM trace collection.

    Features thread-safe context propagation, nested sessions with metadata inheritance,
    and lightweight in-memory buffer. For distributed/shared storage, use OTEL tracing.

    Example:
        >>> with ContextVarSession() as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))
    """

    def __init__(
        self,
        name: str | None = None,
        formatter: Callable[[dict], dict] | None = None,
        persistent_tracers: list | None = None,
        **metadata,
    ):
        """
        Initialize contextvars-based session.

        Args:
            name: Session name (auto-generated if None). If None and there's an
                  existing session name in the context (from a parent session),
                  that will be inherited instead of generating a new one.
            formatter: Optional formatter to transform trace data (deprecated, kept for compatibility)
            persistent_tracers: Optional list of persistent tracers (deprecated, kept for compatibility)
            **metadata: Session metadata
        """
        # If name is not explicitly provided, check if there's one in the context
        # (set by a parent _session_with_name). This allows internal code to control
        # the session name while keeping it hidden from users via the session() shortcut.
        if name is None:
            existing_name = _session_name.get()
            if existing_name is not None:
                name = existing_name

        # Generate new name only if none was provided and none exists in context
        self.name = name or f"sess_{uuid.uuid4().hex[:16]}"

        # Internal unique ID for this session context instance (different from name which can be inherited)
        # This allows tracking each unique session context even when name is shared
        self._uid = f"ctx_{uuid.uuid4().hex[:16]}"

        # Build session UID chain for tree hierarchy support
        parent_session = get_current_cv_session()
        if parent_session is not None:
            # Inherit parent's chain and append our UID
            self._session_uid_chain = parent_session._session_uid_chain + [self._uid]
        else:
            # Root session - start new chain
            self._session_uid_chain = [self._uid]

        self.metadata = metadata
        self.formatter = formatter or (lambda x: x)

        # Each session gets its own lightweight in-memory buffer
        # (distributed/shared storage is handled by OTEL, not ContextVarSession)
        self.storage = SessionBuffer()

        # Optional persistent tracers (kept for backward compatibility)
        self._persistent_tracers = persistent_tracers or []

        # Context tokens for cleanup
        self._session_token = None
        self._session_name_token = None
        self._metadata_token = None
        self._stack_token = None

    @property
    def llm_calls(self) -> list[Trace]:
        """Get all LLM traces from this session and nested child sessions.

        Parent sessions automatically see traces from nested children via session UID hierarchy.
        """
        return self.storage.get_traces(self._uid, self.name)

    @property
    def steps(self) -> list[StepView]:
        """Get all steps within this session."""
        return [trace_to_step_view(trace) for trace in self.llm_calls]

    def clear_calls(self) -> None:
        """Clear all traces for this session (SessionBuffer only)."""
        if hasattr(self.storage, "clear"):
            self.storage.clear(self._uid, self.name)

    def __enter__(self):
        """Enter session context - set up context variables."""
        # Set this session instance in context
        self._session_token = _current_session.set(self)

        # Set session name in context
        self._session_name_token = _session_name.set(self.name)

        # Merge and set metadata (inherits from parent)
        parent_meta = _metadata.get() or {}
        merged = {**parent_meta, **self.metadata}
        self._metadata_token = _metadata.set(merged)

        # Push onto sessions stack (outer → inner order)
        current_stack = _sessions_stack.get() or []
        # Create a new list to avoid mutating shared references
        new_stack = list(current_stack)
        new_stack.append(self)
        self._stack_token = _sessions_stack.set(new_stack)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - restore previous context."""
        if self._session_token is not None:
            _current_session.reset(self._session_token)
        if self._session_name_token is not None:
            _session_name.reset(self._session_name_token)
        if self._metadata_token is not None:
            _metadata.reset(self._metadata_token)

        # Restore previous sessions stack
        if self._stack_token is not None:
            _sessions_stack.reset(self._stack_token)

        return False

    def __len__(self) -> int:
        """Return number of calls in this session."""
        return len(self.llm_calls)

    def __repr__(self):
        return f"ContextVarSession(name={self.name!r}, _uid={self._uid!r}, chain_depth={len(self._session_uid_chain)}, storage={self.storage!r})"
