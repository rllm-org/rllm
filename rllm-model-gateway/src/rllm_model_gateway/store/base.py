"""TraceStore protocol — the abstract interface for trace persistence."""

from typing import Any, Protocol


class TraceStore(Protocol):
    """Abstract storage backend for trace persistence.

    Implementations must be async.  The interface uses plain dicts so that
    backends are free to serialise however they like (JSON columns, DynamoDB
    items, etc.).

    Traces are stamped with ``run_id`` so multiple gateway instances can
    share a single store without colliding session ids. ``run_id=""`` is
    the unstamped bucket for legacy callers.
    """

    async def store_trace(
        self,
        trace_id: str,
        session_id: str,
        data: dict[str, Any],
        run_id: str = "",
    ) -> None:
        """Store a single trace, optionally tagged with ``run_id``."""
        ...

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by ID.  Returns ``None`` when not found."""
        ...

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all traces for a session, ordered by timestamp ascending.

        ``run_id=None`` is cross-run; pass an explicit value to filter.
        """
        ...

    async def delete_session(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> int:
        """Delete a session.

        ``run_id=None`` deletes every match across runs. Returns the count
        of trace rows deleted (rows referenced from other sessions are
        kept).
        """
        ...

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions with trace counts. Each row carries ``run_id``."""
        ...

    async def register_run(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record (or update) a gateway run's metadata.

        Idempotent: a re-call with the same ``run_id`` updates metadata
        in place but preserves the original ``started_at``.
        """
        ...

    async def end_run(self, run_id: str) -> None:
        """Mark a run as ended (sets ``ended_at`` if currently null)."""
        ...

    async def list_runs(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List gateway runs ordered by ``started_at`` DESC."""
        ...

    async def flush(self) -> None:
        """Flush any buffered writes to durable storage."""
        ...

    async def close(self) -> None:
        """Release any resources held by the store."""
        ...
