"""TraceStore protocol — abstract interface for trace persistence."""

from __future__ import annotations

from typing import Any, Protocol

from rllm_model_gateway.trace import TraceRecord


class SessionInfo(dict):
    """Plain dict alias for session metadata returned by the store."""


class TraceStore(Protocol):
    # -- Sessions --------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a session row."""
        ...

    async def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return sessions with their trace counts."""
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete a session and cascade its traces. Returns trace count deleted."""
        ...

    # -- Traces ----------------------------------------------------------

    async def store_trace(
        self,
        trace: TraceRecord,
        extras: tuple[str, bytes] | None = None,
    ) -> None:
        """Persist a trace + optional extras blob.

        ``extras`` is ``(format, bytes)`` where format is e.g. "msgpack".
        """
        ...

    async def get_trace(self, trace_id: str, extras: bool = False) -> TraceRecord | None:
        """Return a trace. ``extras=False`` leaves the field None; ``extras=True``
        populates it from the trace_extras table (or ``{}`` if empty)."""
        ...

    async def get_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        extras: bool = False,
    ) -> list[TraceRecord]:
        """Return all traces for a session, ordered by timestamp asc.

        ``extras=False`` leaves the field None on each row; ``extras=True``
        populates it from the trace_extras table.
        """
        ...

    # -- Lifecycle -------------------------------------------------------

    async def flush(self) -> None: ...
    async def close(self) -> None: ...
