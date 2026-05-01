"""``mount_console`` — attach the console UI to a FastAPI app."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rllm.console.panels import list_panels

logger = logging.getLogger(__name__)


# Built-in panels imported here so they self-register on first
# ``mount_console`` call. New panels added under ``rllm.console.panels.*``
# only need an entry here.
_DEFAULT_PANEL_MODULES: tuple[str, ...] = (
    "rllm.console.panels.sessions",
    "rllm.console.panels.runs",
    "rllm.console.panels.sandboxes",
    "rllm.console.panels.eval_launcher",
    "rllm.console.panels.training",
    "rllm.console.panels.settings",
)


def _import_default_panels() -> None:
    for module_name in _DEFAULT_PANEL_MODULES:
        try:
            import_module(module_name)
        except Exception:
            logger.exception("rllm-console: failed to load panel %s", module_name)


def default_static_dir() -> Path:
    """Where the bundled frontend lands after ``rllm view --build``."""
    return Path(__file__).parent / "static"


def mount_console(
    app: FastAPI,
    *,
    eval_results_root: Path | None = None,
    static_dir: Path | None = None,
    url_prefix: str = "/console",
    version: str = "0.1.0",
) -> None:
    """Attach the rLLM Console to an existing FastAPI app.

    Args:
        app: FastAPI application (typically the model gateway) to mount onto.
        eval_results_root: Directory the Runs panel reads from. ``None``
            disables run-scoped panels (Sessions still works against the
            gateway's TraceStore).
        static_dir: Override the built-frontend directory. Defaults to
            ``rllm/console/static`` shipped in the wheel.
        url_prefix: Path prefix for everything the console adds. Defaults
            to ``/console``. Must not end with ``/``.
        version: Version string surfaced via ``/api/shell/info``.
    """
    if url_prefix.endswith("/"):
        url_prefix = url_prefix.rstrip("/")
    if not url_prefix.startswith("/"):
        url_prefix = "/" + url_prefix

    _import_default_panels()
    panels = list_panels()

    resolved_root: Path | None = None
    if eval_results_root is not None:
        resolved_root = Path(eval_results_root).expanduser().resolve()

    # Per-app state so panel routes can read the configured root without
    # closing over function args.
    app.state.console_eval_results_root = resolved_root
    app.state.console_url_prefix = url_prefix
    app.state.console_version = version

    # ---- Shell info (always first, never matched by SPA catch-all) -----
    shell = APIRouter()

    @shell.get("/api/shell/info")
    def _shell_info() -> dict:
        return {
            "version": version,
            "url_prefix": url_prefix,
            "eval_results_root": (str(resolved_root) if resolved_root is not None else None),
            "panels": [
                {
                    "id": p.id,
                    "title": p.title,
                    "icon": p.icon,
                    "nav_order": p.nav_order,
                    "placeholder": p.placeholder,
                    "has_router": p.router is not None,
                }
                for p in panels
            ],
        }

    app.include_router(shell, prefix=url_prefix)

    # ---- Per-panel routers ---------------------------------------------
    for panel in panels:
        if panel.router is None:
            continue
        app.include_router(
            panel.router,
            prefix=f"{url_prefix}/api/panels/{panel.id}",
            tags=[f"console:{panel.id}"],
        )

    # ---- Static frontend (last so panel routes win) --------------------
    static = static_dir or default_static_dir()
    if static.is_dir():
        _mount_static(app, static, url_prefix=url_prefix)
    else:
        logger.info(
            "rllm-console: no built frontend at %s — API only. Run `rllm view --build` to build the SPA.",
            static,
        )


def _mount_static(app: FastAPI, static_dir: Path, *, url_prefix: str) -> None:
    """Wire up the built SPA bundle under ``url_prefix``."""
    assets_dir = static_dir / "assets"
    if assets_dir.is_dir():
        app.mount(
            f"{url_prefix}/assets",
            StaticFiles(directory=assets_dir),
            name="console_assets",
        )

    fonts_dir = static_dir / "fonts"
    if fonts_dir.is_dir():
        app.mount(
            f"{url_prefix}/fonts",
            StaticFiles(directory=fonts_dir),
            name="console_fonts",
        )

    favicon_path = static_dir / "favicon.ico"

    @app.get(f"{url_prefix}/favicon.ico", include_in_schema=False)
    def _console_favicon() -> FileResponse:
        if not favicon_path.is_file():
            raise HTTPException(404)
        return FileResponse(favicon_path)

    index_path = static_dir / "index.html"
    static_root = static_dir.resolve()

    @app.get(url_prefix, include_in_schema=False)
    def _console_root() -> FileResponse:
        return FileResponse(index_path)

    @app.get(url_prefix + "/{path:path}", include_in_schema=False)
    def _console_spa(path: str):
        # API paths must not be hijacked by the SPA fallback — an
        # unmatched /console/api/... has to surface as a 404 so panel
        # bugs don't silently return HTML and confuse clients. Only the
        # exact `api` literal is excluded; SPA paths like `apiary` or
        # `apparel` (unlikely but safe) still pass through.
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404)
        candidate = (static_dir / path).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            return FileResponse(index_path)
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)
