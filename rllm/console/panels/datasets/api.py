"""Datasets panel API.

* ``GET /``                                — every dataset as a card row.
* ``GET /categories``                      — category names for filter pills.
* ``GET /{name}``                          — full detail incl. per-split counts.
* ``GET /{name}/entries?split=&offset=&limit=`` — paginated row reader.
* ``POST /{name}/pull``                    — SSE-streamed subprocess wrapping
                                              ``rllm dataset pull <name>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from rllm.console.panels.datasets import loader

logger = logging.getLogger(__name__)

router = APIRouter()

# Permits the ``harbor:`` prefix so the user can pull
# ``harbor:swebench-verified`` from the UI. Colon is *not* a shell
# metacharacter when commands run via argv (no shell=True), but the
# regex still excludes ``;``, ``|``, ``&``, ``$``, backticks, etc.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._:-]+$")


def _validate_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "Bad dataset name")
    return name


@router.get("")
@router.get("/")
def list_datasets() -> dict[str, Any]:
    return {
        "datasets": loader.list_datasets(),
        "categories": loader.categories(),
    }


@router.get("/categories")
def list_categories() -> list[str]:
    return loader.categories()


@router.get("/{name}")
def get_dataset(name: str) -> dict[str, Any]:
    _validate_name(name)
    detail = loader.get_dataset(name)
    if detail is None:
        raise HTTPException(404, f"unknown dataset: {name}")
    return detail


@router.get("/{name}/entries")
def get_entries(
    name: str,
    split: str = Query(..., min_length=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    _validate_name(name)
    if not _SAFE_NAME.match(split):
        raise HTTPException(400, "Bad split name")
    try:
        return loader.get_entries(name, split=split, offset=offset, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


# ---------------------------------------------------------------------------
# Pull (subprocess + SSE)
# ---------------------------------------------------------------------------


@router.post("/{name}/pull")
async def pull_dataset(name: str) -> StreamingResponse:
    """Spawn ``rllm dataset pull <name>`` and stream its output.

    Each subprocess line lands as one SSE ``data:`` event with a
    ``{type, line}`` payload. Three structured events bracket the
    stream so the UI can drive its own state machine without having
    to parse log text:

    * ``{"type":"start","name":...}``           — first frame
    * ``{"type":"log","line":"..."}``           — every output line
    * ``{"type":"done","ok":bool,"exit_code":N}`` — final frame

    The endpoint is a plain POST (not loopback-guarded) — pulling a
    dataset is a write to the local cache, not an exec primitive
    against a sandbox. Name validation is the only injection guard.
    """
    _validate_name(name)
    rllm_bin = shutil.which("rllm")
    if rllm_bin is None:
        raise HTTPException(503, "the `rllm` CLI isn't on PATH")

    return StreamingResponse(
        _stream_pull(rllm_bin, name),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


async def _stream_pull(rllm_bin: str, name: str) -> AsyncIterator[str]:
    yield _sse({"type": "start", "name": name})
    proc = await asyncio.create_subprocess_exec(
        rllm_bin,
        "dataset",
        "pull",
        name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield _sse({"type": "log", "line": line.decode("utf-8", errors="replace").rstrip("\n")})
        await proc.wait()
        yield _sse({"type": "done", "ok": proc.returncode == 0, "exit_code": proc.returncode or 0})
    except asyncio.CancelledError:
        # Client closed the connection — best-effort kill the child so
        # we don't leak a half-finished pull.
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=2.0)
        raise
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
