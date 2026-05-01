"""Sessions panel — cross-run gateway tracing browser.

Three endpoints, all read-only against the shared sqlite trace store:

* ``GET /runs``                   — runs with derived session/trace counts
* ``GET /sessions?run_id=``       — session summaries (per-run or cross-run)
* ``GET /traces?session_id=&run_id=&since=&limit=`` — trace timeline for one session

Ported from ``rllm/eval/visualizer.py``'s ``/api/gateway/*`` handlers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from rllm.eval import trace_loader

router = APIRouter()


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """Every gateway run with observed session and trace counts.

    Includes orphan runs (registered but no traces yet) and "ghost" runs
    (sessions whose run was never registered, e.g. legacy schema rows).
    Sorted by most-recent activity first.
    """
    db = trace_loader.default_db_path()
    runs = trace_loader.list_runs(db)
    summaries = trace_loader.session_summaries_by_run(db)

    counts: dict[str, int] = {}
    trace_counts: dict[str, int] = {}
    last_at: dict[str, float] = {}
    for s in summaries:
        rid = s["run_id"]
        counts[rid] = counts.get(rid, 0) + 1
        trace_counts[rid] = trace_counts.get(rid, 0) + int(s.get("trace_count") or 0)
        if s.get("last_at") is not None:
            last_at[rid] = max(last_at.get(rid, 0.0), float(s["last_at"]))

    seen_run_ids = {r["run_id"] for r in runs}
    for rid in counts.keys() - seen_run_ids:
        runs.append(
            {
                "run_id": rid,
                "started_at": None,
                "ended_at": None,
                "metadata": {},
            }
        )

    out: list[dict[str, Any]] = []
    for r in runs:
        rid = r["run_id"]
        out.append(
            {
                **r,
                "session_count": counts.get(rid, 0),
                "trace_count": trace_counts.get(rid, 0),
                "last_trace_at": last_at.get(rid),
            }
        )

    out.sort(key=lambda r: -(r.get("last_trace_at") or r.get("started_at") or 0.0))
    return out


@router.get("/sessions")
def list_sessions(
    run_id: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Session summaries.

    With ``run_id`` set, returns sessions for that run. Without it,
    returns every ``(run_id, session_id)`` pair across the store.
    """
    db = trace_loader.default_db_path()
    if run_id:
        summaries = trace_loader.session_summaries(db, run_id=run_id)
        rows = [
            {
                "session_id": sid,
                "run_id": run_id,
                "trace_count": s["trace_count"],
                "first_at": s["first_at"],
                "last_at": s["last_at"],
            }
            for sid, s in summaries.items()
        ]
    else:
        rows = trace_loader.session_summaries_by_run(db)

    rows.sort(key=lambda r: -(r.get("last_at") or 0.0))
    return rows


@router.get("/traces")
def get_traces(
    session_id: str = Query(..., min_length=1),
    run_id: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=10_000),
) -> list[dict[str, Any]]:
    """Traces for a session, ordered by store-time ASC.

    ``since`` is a wall-clock cursor (seconds since epoch); pass the
    last ``_created_at`` you saw to fetch only newer rows for live tail.
    """
    if not session_id:
        raise HTTPException(400, "session_id required")
    return trace_loader.get_traces(
        trace_loader.default_db_path(),
        session_id,
        since=since,
        limit=limit,
        run_id=run_id,
    )
