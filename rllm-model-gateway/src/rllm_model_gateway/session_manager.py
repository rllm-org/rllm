"""Session lifecycle management.

Sessions are created *implicitly* on first request to
``/sessions/{sid}/v1/chat/completions``.  An explicit ``POST /sessions``
endpoint is available for pre-creating sessions with metadata.
"""

import time
import uuid
from typing import Any

from rllm_model_gateway.models import SessionInfo
from rllm_model_gateway.store.base import TraceStore


class SessionManager:
    """Thin wrapper around ``TraceStore`` for session-level operations."""

    def __init__(self, store: TraceStore) -> None:
        self.store = store
        # In-memory metadata cache (not persisted — metadata is lightweight)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._created_at: dict[str, float] = {}

    def ensure_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Ensure a session exists (create if needed).  Returns session_id."""
        if session_id not in self._created_at:
            self._created_at[session_id] = time.time()
        if metadata:
            self._metadata.setdefault(session_id, {}).update(metadata)
        return session_id

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new session and return its ID."""
        sid = session_id or str(uuid.uuid4())
        self._created_at[sid] = time.time()
        if metadata:
            self._metadata[sid] = metadata
        return sid

    async def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Return session info including trace count."""
        traces = await self.store.get_session_traces(session_id)
        if not traces and session_id not in self._created_at:
            return None
        return SessionInfo(
            session_id=session_id,
            trace_count=len(traces),
            created_at=self._created_at.get(session_id),
            metadata=self._metadata.get(session_id, {}),
        )

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[SessionInfo]:
        """List all sessions from the store."""
        rows = await self.store.list_sessions(since=since, limit=limit)
        results: list[SessionInfo] = []
        for row in rows:
            sid = row["session_id"]
            results.append(
                SessionInfo(
                    session_id=sid,
                    trace_count=row.get("trace_count", 0),
                    created_at=self._created_at.get(sid, row.get("first_trace_at")),
                    metadata=self._metadata.get(sid, {}),
                )
            )
        return results

    async def delete_session(self, session_id: str) -> int:
        """Delete a session and its traces.  Returns count of traces deleted."""
        self._metadata.pop(session_id, None)
        self._created_at.pop(session_id, None)
        return await self.store.delete_session(session_id)
