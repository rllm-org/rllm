"""Runs panel — eval-results filesystem + gateway-runs union.

* ``GET /``                                      — run summaries (disk + gateway)
* ``GET /{run_id}/index``                        — episode index (disk-only)
* ``GET /{run_id}/episodes/{filename}``          — one episode JSON (disk-only)
* ``GET /{run_id}/live``                         — liveness snapshot
* ``GET /{run_id}/traces?<filters>``             — paginated trace feed for this run

Path safety mirrors the visualizer: run ids are validated against a
strict regex, and resolved targets must stay inside the configured root.
Gateway-only runs (no disk dir yet) get 404 on disk-bound endpoints
but are reachable via ``/{run_id}/live`` and ``/{run_id}/traces``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from rllm.console.panels.runs import loader
from rllm.eval import trace_loader

router = APIRouter()

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_EPISODE_FILE = re.compile(r"^episode_[A-Za-z0-9._-]+\.json$")


def _eval_results_root(request: Request) -> Path:
    root = getattr(request.app.state, "console_eval_results_root", None)
    if root is None:
        raise HTTPException(503, "console eval_results_root not configured")
    return root


def _check_run_id(run_id: str) -> None:
    """Reject path-traversing or otherwise unsafe run ids."""
    if not run_id or run_id in (".", "..") or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(400, "Bad run id")
    if not _SAFE_RUN_ID.match(run_id):
        raise HTTPException(400, "Bad run id")


def _validate_run_dir(root: Path, run_id: str) -> Path:
    """Disk-bound endpoints: require a real episodes dir under root."""
    _check_run_id(run_id)
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "Bad run id") from None
    if not run_dir.is_dir():
        raise HTTPException(404, "Run not found")
    return run_dir


def _resolve_run_dir_or_synthetic(root: Path, run_id: str) -> Path:
    """Return a disk run dir if it exists, otherwise a synthetic path.

    Used by liveness and trace endpoints, which are valid for
    gateway-only runs that haven't materialised on disk.
    """
    _check_run_id(run_id)
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "Bad run id") from None
    return run_dir


@router.get("")
@router.get("/")
def list_runs(request: Request) -> list[dict[str, Any]]:
    """Run summaries — union of disk + gateway, ordered by start time DESC."""
    root = _eval_results_root(request)
    return loader.scan_runs(root)


@router.get("/{run_id}/index")
def episode_index(run_id: str, request: Request) -> list[dict[str, Any]]:
    root = _eval_results_root(request)
    run_dir = _validate_run_dir(root, run_id)
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        raise HTTPException(404, "Run not found")
    return loader.build_episode_index(episodes_dir)


@router.get("/{run_id}/episodes/{filename}")
def episode_file(run_id: str, filename: str, request: Request) -> FileResponse:
    root = _eval_results_root(request)
    run_dir = _validate_run_dir(root, run_id)
    if not _SAFE_EPISODE_FILE.match(filename):
        raise HTTPException(400, "Bad path")
    target = (run_dir / "episodes" / filename).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "Bad path") from None
    if not target.is_file():
        raise HTTPException(404, "Episode not found")
    return FileResponse(target, media_type="application/json")


@router.get("/{run_id}/live")
def live_payload(run_id: str, request: Request) -> dict[str, Any]:
    """Liveness + sessions snapshot. Works for gateway-only runs too."""
    root = _eval_results_root(request)
    run_dir = _resolve_run_dir_or_synthetic(root, run_id)
    return loader.build_live_payload(run_dir)


@router.get("/{run_id}/traces")
def run_traces(
    run_id: str,
    request: Request,
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
    """Run-scoped trace feed. Same filter shape as the Sessions panel,
    pinned to this run's gateway ``run_id``.
    """
    root = _eval_results_root(request)
    run_dir = _resolve_run_dir_or_synthetic(root, run_id)
    rid = loader.gateway_run_id(run_dir) if run_dir.is_dir() else run_id
    return trace_loader.query_traces(
        trace_loader.default_db_path(),
        run_id=rid,
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
