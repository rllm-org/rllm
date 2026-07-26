"""Typer CLI: `rllm-trace-uploader oneshot | daemon`."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import typer

from rllm_trace_uploader.sidecar import SidecarReader
from rllm_trace_uploader.uploader import TraceUploader

app = typer.Typer(help="Republish gateway SQLite traces to W&B Weave.")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rllm-trace-uploader")


def _default_state_path() -> str:
    root = os.environ.get("RLLM_HOME", "/tmp/rllm_home")
    return os.path.join(root, "trace_uploader_state.txt")


def _default_sidecar_root() -> str:
    root = os.environ.get("RLLM_HOME", "/tmp/rllm_home")
    return os.path.join(root, "observability")


def _build_uploader(
    db_path: str,
    project: str,
    state_path: str,
    sidecar_root: str,
) -> TraceUploader:
    sidecar = SidecarReader(sidecar_root)
    return TraceUploader(
        db_path=db_path,
        state_path=state_path,
        weave_project=project,
        sidecar=sidecar,
    )


@app.command()
def oneshot(
    db_path: str = typer.Option(..., "--db-path", help="Path to gateway traces.db"),
    project: str = typer.Option(..., "--project", help="W&B project (Weave)"),
    state_path: str = typer.Option(_default_state_path(), "--state-path"),
    sidecar_root: str = typer.Option(_default_sidecar_root(), "--sidecar-root"),
    limit: int = typer.Option(1000, "--limit"),
) -> None:
    """Publish new traces once, then exit."""
    up = _build_uploader(db_path, project, state_path, sidecar_root)
    n = asyncio.run(up.oneshot(limit=limit))
    logger.info("Uploaded %d traces to Weave project=%s (last_rowid=%d)", n, project, up._last_rowid)


@app.command()
def daemon(
    db_path: str = typer.Option(..., "--db-path"),
    project: str = typer.Option(..., "--project"),
    state_path: str = typer.Option(_default_state_path(), "--state-path"),
    sidecar_root: str = typer.Option(_default_sidecar_root(), "--sidecar-root"),
    interval: int = typer.Option(60, "--interval", help="Poll interval in seconds"),
    limit: int = typer.Option(1000, "--limit", help="Max traces per poll"),
) -> None:
    """Publish new traces in a loop until interrupted."""
    up = _build_uploader(db_path, project, state_path, sidecar_root)
    logger.info(
        "daemon starting: db=%s project=%s interval=%ds last_rowid=%d",
        db_path,
        project,
        interval,
        up._last_rowid,
    )
    try:
        while True:
            if not os.path.exists(db_path):
                logger.warning("db not found (yet): %s", db_path)
            else:
                try:
                    n = asyncio.run(up.oneshot(limit=limit))
                    if n:
                        logger.info("Uploaded %d traces (last_rowid=%d)", n, up._last_rowid)
                except Exception:
                    logger.exception("upload cycle failed")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("daemon stopping")


if __name__ == "__main__":
    app()
