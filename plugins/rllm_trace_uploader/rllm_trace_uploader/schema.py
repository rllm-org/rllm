"""Row model matching gateway SqliteTraceStore `traces` table.

Schema (from rllm-model-gateway sqlite_store.py):
    traces(trace_id TEXT PRIMARY KEY, data TEXT, created_at REAL)
where `data` is the JSON-serialized TraceRecord.model_dump().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceRow:
    """One SQLite row unpacked from `traces` + `trace_sessions` join."""

    trace_id: str
    session_id: str
    created_at: float
    # From TraceRecord.model_dump():
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    response_message: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_counts: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    # Row primary key from SQLite rowid, used as the resume cursor.
    rowid: int = 0

    @classmethod
    def from_sqlite(
        cls,
        *,
        rowid: int,
        trace_id: str,
        session_id: str,
        data_json: str,
        created_at: float,
    ) -> "TraceRow":
        data: dict[str, Any] = json.loads(data_json) if data_json else {}
        return cls(
            trace_id=trace_id,
            session_id=session_id,
            created_at=created_at,
            model=data.get("model", ""),
            messages=data.get("messages") or [],
            response_message=data.get("response_message") or {},
            latency_ms=float(data.get("latency_ms", 0.0)),
            token_counts=data.get("token_counts") or {},
            finish_reason=data.get("finish_reason"),
            metadata=data.get("metadata") or {},
            raw_request=data.get("raw_request"),
            raw_response=data.get("raw_response"),
            rowid=rowid,
        )
