"""Runs panel — eval-results filesystem browser.

Five endpoints, all rooted at the ``eval_results_root`` configured via
:func:`rllm.console.mount_console`:

* ``GET /``                                     — run summaries
* ``GET /{run_id}/index``                       — episode index
* ``GET /{run_id}/episodes/{filename}``         — one episode JSON
* ``GET /{run_id}/traces?session_id=&...``      — run-scoped trace timeline
* ``GET /{run_id}/live``                        — in-flight task snapshot

Path safety mirrors the visualizer: run ids are validated against a
strict regex, and resolved targets must stay inside the configured root.
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


def _validate_run_dir(root: Path, run_id: str) -> Path:
    if not run_id or run_id in (".", "..") or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(400, "Bad run id")
    if not _SAFE_RUN_ID.match(run_id):
        raise HTTPException(400, "Bad run id")
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "Bad run id") from None
    if not run_dir.is_dir():
        raise HTTPException(404, "Run not found")
    return run_dir


@router.get("")
@router.get("/")
def list_runs(request: Request) -> list[dict[str, Any]]:
    """Run summaries (filesystem scan + aggregate JSON merge)."""
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
    root = _eval_results_root(request)
    run_dir = _validate_run_dir(root, run_id)
    return loader.build_live_payload(run_dir)


@router.get("/{run_id}/traces")
def run_traces(
    run_id: str,
    request: Request,
    session_id: str = Query(..., min_length=1),
    since: float | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=10_000),
) -> list[dict[str, Any]]:
    root = _eval_results_root(request)
    run_dir = _validate_run_dir(root, run_id)
    return trace_loader.get_traces(
        trace_loader.default_db_path(),
        session_id,
        since=since,
        limit=limit,
        run_id=loader.gateway_run_id(run_dir),
    )
