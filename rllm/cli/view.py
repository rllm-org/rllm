"""``rllm view`` — launch the rLLM Console.

Boots a view-only gateway (TraceStore + console UI mounted; no proxy
workers), opens the browser at ``/console/``. The console serves both
the live trace browser and the eval-run grid.

Three modes, mirroring ``harbor view``:

* **production** (default) — serves the pre-built SPA from
  ``rllm/console/static/``. If the dir is missing, attempts to build
  the frontend with ``bun`` from ``rllm-console/``. ``--no-build``
  skips the build and runs API-only.
* **--build** — force a fresh frontend build before starting.
* **--dev** — runs ``bun run dev`` (Vite at :5173) as a subprocess
  alongside uvicorn so frontend edits hot-reload. Backend reloads
  on changes under ``rllm/console/``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

import click
from rich.console import Console

console = Console(stderr=True)


# Repo layout: rllm/cli/view.py → ../../rllm-console/  (frontend source)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _REPO_ROOT / "rllm-console"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "console" / "static"
_FRONTEND_BUILD_OUTPUT = _FRONTEND_DIR / "build" / "client"


def _eval_results_root() -> Path:
    rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
    return Path(rllm_home) / "eval_results"


def _resolve_target(arg: str | None) -> Path:
    """Resolve a CLI arg to an eval-results root.

    * No arg → ``~/.rllm/eval_results/``.
    * Existing dir → its parent if it looks like a single run (has
      ``episodes/``), else itself.
    * Run-id under default root → that run's parent (the default root).
    """
    if arg is None:
        return _eval_results_root()
    p = Path(arg).expanduser()
    if p.is_dir():
        if (p / "episodes").is_dir():
            return p.parent.resolve()
        return p.resolve()
    candidate = _eval_results_root() / arg
    if candidate.is_dir():
        return _eval_results_root()
    raise SystemExit(f"Could not find: {arg!r} (also tried {candidate})")


def _has_bun() -> bool:
    return shutil.which("bun") is not None


def _build_frontend() -> bool:
    """``bun install && bun run build`` in ``rllm-console/`` then copy
    to ``rllm/console/static``. Returns True on success."""
    if not _FRONTEND_DIR.exists():
        console.print(f"[red]Error:[/red] frontend source not found at {_FRONTEND_DIR}")
        return False
    if not _has_bun():
        console.print("[red]Error:[/red] bun is required to build the console. Install from https://bun.com or pass --no-build.")
        return False

    console.print("[blue]Building rllm-console…[/blue]")
    console.print("  installing dependencies…")
    r = subprocess.run(["bun", "install"], cwd=_FRONTEND_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        console.print("[red]Error:[/red] bun install failed")
        console.print(r.stderr)
        return False

    console.print("  building frontend…")
    r = subprocess.run(["bun", "run", "build"], cwd=_FRONTEND_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        console.print("[red]Error:[/red] bun run build failed")
        console.print(r.stderr)
        return False

    if not _FRONTEND_BUILD_OUTPUT.exists():
        console.print(f"[red]Error:[/red] build output missing at {_FRONTEND_BUILD_OUTPUT}")
        return False

    if _STATIC_DIR.exists():
        shutil.rmtree(_STATIC_DIR)
    shutil.copytree(_FRONTEND_BUILD_OUTPUT, _STATIC_DIR)
    console.print("[green]Build complete.[/green]")
    return True


def _parse_port_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    p = int(spec)
    return p, p


def _find_available_port(host: str, start: int, end: int) -> int | None:
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue
    return None


@click.command("view")
@click.argument("target", required=False)
@click.option("--port", "port_spec", default="7860-7869", help="Port or range (e.g. 7860 or 7860-7869).")
@click.option("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1.")
@click.option("--dev", is_flag=True, help="Run frontend in dev mode (bun + Vite hot reload).")
@click.option("--build", "force_build", is_flag=True, help="Force rebuild of the frontend before starting.")
@click.option("--no-build", "no_build", is_flag=True, help="Skip auto-building the frontend if static files are missing.")
@click.option("--no-browser", "no_browser", is_flag=True, help="Do not auto-open the browser.")
def view_cmd(
    target: str | None,
    port_spec: str,
    host: str,
    dev: bool,
    force_build: bool,
    no_build: bool,
    no_browser: bool,
) -> None:
    """Launch the rLLM Console (live tracing + eval-run browser)."""
    eval_root = _resolve_target(target)

    start_port, end_port = _parse_port_range(port_spec)
    port = _find_available_port(host, start_port, end_port)
    if port is None:
        console.print(f"[red]Error:[/red] no available port in range {start_port}-{end_port}")
        raise SystemExit(1)

    if dev:
        if force_build:
            console.print("[yellow]Warning:[/yellow] --build is ignored in --dev mode.")
        _run_dev(host, port, eval_root, no_browser=no_browser)
    else:
        _run_production(host, port, eval_root, force_build=force_build, no_build=no_build, no_browser=no_browser)


def _run_production(
    host: str,
    port: int,
    eval_root: Path,
    *,
    force_build: bool,
    no_build: bool,
    no_browser: bool,
) -> None:
    """Single-process: uvicorn serving the gateway+console with built static."""
    static_dir: Path | None = _STATIC_DIR if _STATIC_DIR.is_dir() else None

    if force_build:
        if not _build_frontend():
            raise SystemExit(1)
        static_dir = _STATIC_DIR
    elif static_dir is None and not no_build and _FRONTEND_DIR.exists():
        console.print("[yellow]Console not built — building now (one-time).[/yellow]")
        if _build_frontend():
            static_dir = _STATIC_DIR
        else:
            console.print("[yellow]Build failed; serving API only.[/yellow]")
            console.print("  Try [bold]rllm view --dev[/bold] for hot-reload, or pass --no-build.")

    if static_dir is None and not no_build:
        console.print("[yellow]Warning:[/yellow] frontend not built. API only.")

    app = _build_app(eval_root, static_dir)

    url = f"http://{host}:{port}/console/"
    console.print(f"[green]rLLM Console[/green]   eval root: {eval_root}")
    console.print(f"  serving at {url}")
    console.print("  press Ctrl+C to stop.")

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def _run_dev(host: str, port: int, eval_root: Path, *, no_browser: bool) -> None:
    """Two processes: bun (Vite at :5173) + uvicorn (backend at chosen port)."""
    if not _FRONTEND_DIR.exists():
        console.print(f"[red]Error:[/red] frontend source missing at {_FRONTEND_DIR}")
        raise SystemExit(1)
    if not _has_bun():
        console.print("[red]Error:[/red] bun is required for --dev mode (install from https://bun.com).")
        raise SystemExit(1)

    console.print(f"[green]rLLM Console (dev)[/green]   eval root: {eval_root}")
    console.print(f"  backend  http://{host}:{port}/console/api/...")
    console.print("  frontend http://localhost:5173/console/   (hot reload)")

    # Frontend deps + dev server.
    console.print("  installing frontend deps…")
    r = subprocess.run(["bun", "install"], cwd=_FRONTEND_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        console.print(f"[red]bun install failed:[/red]\n{r.stderr}")
        raise SystemExit(1)

    env = os.environ.copy()
    env["VITE_API_URL"] = f"http://{host}:{port}"
    frontend = subprocess.Popen(["bun", "run", "dev"], cwd=_FRONTEND_DIR, env=env)

    # Backend: build app and run uvicorn (no reload — the eval_root and
    # static state don't change here, and reload-on-watch needs an
    # import string rather than an instance).
    app = _build_app(eval_root, None)

    if not no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5173/console/")).start()

    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        frontend.terminate()
        try:
            frontend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend.kill()


def _build_app(eval_root: Path, static_dir: Path | None):
    """Compose: gateway (view-only) + console mount."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from rllm.console import mount_console
    from rllm.console.panels.settings import store as settings_store

    # Load any user-managed env vars from ~/.rllm/console.env *before*
    # the gateway/console initialise — the trace store path and
    # provider keys both read os.environ at construction time.
    loaded = settings_store.load_into_environ()
    if loaded:
        console.print(f"  loaded {len(loaded)} env var(s) from {settings_store.default_env_path()}")

    # View-only: a bare FastAPI app, no proxy workers, no model routes.
    # The console only needs the trace store (read-side via trace_loader)
    # and filesystem access (eval_results_root). Gateway features can be
    # composed later by callers that already have a configured gateway
    # FastAPI instance.
    app = FastAPI(title="rllm-console", version="0.1.0")

    # Permissive CORS for `--dev` mode: bun dev server (5173) calls
    # backend (7860) and the proxy is in-place, but explicit headers
    # don't hurt and let downstream tools call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    mount_console(app, eval_results_root=eval_root, static_dir=static_dir)
    return app
