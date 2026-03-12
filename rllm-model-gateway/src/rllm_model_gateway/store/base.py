"""TraceStore protocol — the abstract interface for trace persistence."""

from typing import Any, Protocol


class TraceStore(Protocol):
    """Abstract storage backend for trace persistence.

    Implementations must be async.  The interface uses plain dicts so that
    backends are free to serialise however they like (JSON columns, DynamoDB
    items, etc.).
    """

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
        """Store a single trace."""
        ...

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by ID.  Returns ``None`` when not found."""
        ...

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all traces for a session, ordered by timestamp ascending."""
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all traces for a session.  Returns count deleted."""
        ...

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions with trace counts."""
        ...

    async def flush(self) -> None:
        """Flush any buffered writes to durable storage."""
        ...

    async def close(self) -> None:
        """Release any resources held by the store."""
        ...
