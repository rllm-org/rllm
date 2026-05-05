"""Sessions panel — global gateway-trace feed with filter pushdown.

Two endpoints, both read-only against the shared sqlite trace store:

* ``GET /traces?<filters>`` — paginated trace feed (newest first by default).
  All filters are optional and pushed down to SQL via the denormalized
  columns on ``traces`` (``run_id``, ``session_id``, ``model``,
  ``harness``, ``has_error``, ``latency_ms``, ``created_at``).
* ``GET /facets``           — distinct values for the filter-bar dropdowns.

Cursor pagination uses ``until`` (older-than, for scroll-back) and
``since`` (newer-than, for live tail) on persisted-at time. Pass the
last row's ``_created_at`` back to advance the cursor.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from rllm.eval import trace_loader

router = APIRouter()


@router.get("/traces")
def query_traces(
    run_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    model: str | None = Query(default=None),
    harness: str | None = Query(default=None),
    has_error: bool | None = Query(default=None),
    latency_min: float | None = Query(default=None, ge=0),
    latency_max: float | None = Query(default=None, ge=0),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    order: str = Query(default="DESC"),
) -> list[dict[str, Any]]:
    """Filter+paginate the global trace feed.

    Default order is DESC (newest first) for the global feed; pass
    ``order=ASC`` for per-session timelines.
    """
    return trace_loader.query_traces(
        trace_loader.default_db_path(),
        run_id=run_id,
        session_id=session_id,
        model=model,
        harness=harness,
        has_error=has_error,
        latency_min=latency_min,
        latency_max=latency_max,
        since=since,
        until=until,
        limit=limit,
        order=order,
    )


@router.get("/facets")
def list_facets() -> dict[str, list[str]]:
    """Distinct values for filter-bar dropdowns: ``models``, ``harnesses``, ``runs``."""
    return trace_loader.list_facets(trace_loader.default_db_path())
