"""Base protocol for session implementations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionProtocol(Protocol):
    """
    Common interface for all session implementations.

    All session types (ContextVar-based, OpenTelemetry-based, etc.)
    must implement this protocol.
    """

    name: str
    metadata: dict[str, Any]

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        """Get all LLM calls made within this session."""
        ...

    def clear_calls(self) -> None:
        """Clear all stored calls for this session."""
        ...

    def __enter__(self) -> "SessionProtocol":
        """Enter session context."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit session context."""
        ...
