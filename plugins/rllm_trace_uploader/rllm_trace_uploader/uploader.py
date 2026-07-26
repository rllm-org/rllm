"""Poll SQLite `traces` table and republish each row to W&B Weave.

Usage:
    up = TraceUploader(db_path=..., state_path=..., weave_project="mmcodex-24h", sidecar=...)
    await up.fetch_new_traces()          # generator of TraceRow
    up.publish(row)                      # sync: calls weave op
    up.save_state()                      # atomic write last_pushed_rowid
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiosqlite

from rllm_trace_uploader.schema import TraceRow
from rllm_trace_uploader.sidecar import SidecarReader

logger = logging.getLogger(__name__)

# Weave is imported lazily by _get_publish_fn so unit tests that never publish
# don't need weave installed.
_publish_fn: Any = None


def _get_publish_fn():
    """Import weave lazily and cache the @weave.op-wrapped publisher."""
    global _publish_fn
    if _publish_fn is not None:
        return _publish_fn
    import weave  # noqa: PLC0415  (lazy on purpose)

    @weave.op(name="codex_llm_call")  # type: ignore[misc]
    def _publish_trace(
        session_id: str,
        request: dict[str, Any],
        response: dict[str, Any],
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        return {"session_id": session_id, "response": response, "attrs": attrs}

    _publish_fn = _publish_trace
    return _publish_fn


class TraceUploader:
    def __init__(
        self,
        *,
        db_path: str,
        state_path: str,
        weave_project: str,
        sidecar: SidecarReader,
    ) -> None:
        self.db_path = db_path
        self.state_path = state_path
        self.weave_project = weave_project
        self.sidecar = sidecar
        self._last_rowid = self._load_state()
        self._weave_initialized = False

    def _load_state(self) -> int:
        try:
            with open(self.state_path) as f:
                return int(f.read().strip() or 0)
        except (FileNotFoundError, ValueError):
            return 0

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(self._last_rowid))
        os.replace(tmp, self.state_path)

    def _ensure_weave(self) -> None:
        if not self._weave_initialized:
            import weave  # noqa: PLC0415  (lazy)

            weave.init(self.weave_project)
            self._weave_initialized = True

    async def fetch_new_traces(self, limit: int = 1000) -> list[TraceRow]:
        rows: list[TraceRow] = []
        async with aiosqlite.connect(self.db_path) as conn:
            sql = """
                SELECT t.rowid, t.trace_id, t.data, t.created_at,
                       COALESCE(ts.session_id, '')
                FROM traces t
                LEFT JOIN trace_sessions ts ON ts.trace_id = t.trace_id
                WHERE t.rowid > ?
                ORDER BY t.rowid ASC
                LIMIT ?
            """
            async with conn.execute(sql, (self._last_rowid, limit)) as cur:
                async for row in cur:
                    rowid, trace_id, data_json, created_at, session_id = row
                    rows.append(
                        TraceRow.from_sqlite(
                            rowid=int(rowid),
                            trace_id=trace_id,
                            session_id=session_id or "",
                            data_json=data_json,
                            created_at=float(created_at),
                        )
                    )
        return rows

    def _build_attrs(self, row: TraceRow) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "trace_id": row.trace_id,
            "session_id": row.session_id,
            "model": row.model,
            "latency_ms": row.latency_ms,
            "finish_reason": row.finish_reason,
            "created_at": row.created_at,
            "token_counts": row.token_counts,
        }
        run_id = self.sidecar.get_run_id()
        if run_id:
            attrs["wandb_run_id"] = run_id
        step = self.sidecar.get_current_step()
        if step is not None:
            attrs["training_step"] = step
        ckpt = self.sidecar.get_latest_checkpoint()
        if ckpt:
            attrs["latest_checkpoint"] = ckpt
        return attrs

    def publish(self, row: TraceRow) -> None:
        self._ensure_weave()
        import weave  # noqa: PLC0415  (lazy)

        publish_fn = _get_publish_fn()
        attrs = self._build_attrs(row)
        request = {"messages": row.messages, "raw_request": row.raw_request}
        response = {
            "message": row.response_message,
            "raw_response": row.raw_response,
        }
        with weave.attributes(attrs):
            publish_fn(row.session_id, request, response, attrs)

    async def publish_batch(self, rows: list[TraceRow]) -> int:
        if not rows:
            return 0
        count = 0
        for row in rows:
            try:
                self.publish(row)
                count += 1
            except Exception:  # noqa: BLE001
                logger.exception("publish failed for trace_id=%s", row.trace_id)
                continue
            self._last_rowid = max(self._last_rowid, row.rowid)
        self.save_state()
        return count

    async def oneshot(self, limit: int = 1000) -> int:
        rows = await self.fetch_new_traces(limit=limit)
        return await self.publish_batch(rows)
