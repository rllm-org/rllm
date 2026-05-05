"""``rllm view`` — launch the rLLM episode dashboard.

Usage::

    rllm view                  # browse all runs in ~/.rllm/eval_results/
    rllm view <run_dir>        # drill into a specific run on load
    rllm view <run_id>         # same, looked up under ~/.rllm/eval_results/

The dashboard is served by the Python standard library
(:mod:`http.server`) — no extra dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _eval_results_root() -> Path:
    rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
    return Path(rllm_home) / "eval_results"


def _resolve_target(arg: str | None) -> Path | None:
    """Resolve ``arg`` to a path passable to :func:`visualizer.launch`.

    Returns ``None`` when no arg was given (caller serves the default
    ``~/.rllm/eval_results/`` root).
    """
    if arg is None:
        return None

    p = Path(arg).expanduser()
    if p.exists():
        return p.resolve()

    # Fallback: arg is a run-id under the default root.
    candidate = _eval_results_root() / arg
    if candidate.is_dir():
        return candidate.resolve()

    raise SystemExit(f"Could not find: {arg!r} (also tried {candidate})")


@click.command("view")
@click.argument("target", required=False)
@click.option("--port", "server_port", default=7860, type=int, help="Local port to serve the viewer on (0 = pick free port).")
@click.option("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1 (loopback only).")
@click.option("--no-browser", "no_browser", is_flag=True, default=False, help="Do not auto-open the browser.")
def view_cmd(target: str | None, server_port: int, host: str, no_browser: bool):
    """Browse saved eval episodes in a local web viewer."""
    resolved = _resolve_target(target)
    if resolved is not None:
        console.print(f"  [dim]Target:[/] [bold]{resolved}[/]")
    else:
        console.print(f"  [dim]Browsing all runs under[/] [bold]{_eval_results_root()}[/]")

    from rllm.eval.visualizer import launch

    launch(resolved, server_port=server_port, host=host, open_browser=not no_browser)
