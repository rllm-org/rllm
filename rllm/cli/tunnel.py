"""rLLM CLI: set up and run the public tunnel that lets remote sandboxes reach the gateway.

``rllm tunnel setup`` records backend settings in ``~/.rllm/config.json``
(per-user, so nothing personal lands in shared train scripts). ``rllm tunnel up``
runs that backend as a background daemon and writes its live URL to a state file;
set ``RLLM_GATEWAY_TUNNEL`` to reuse that URL. Wildcard setups instead create
an owned tunnel per eval/train run.
"""

from __future__ import annotations

import subprocess

import click

# Gateway port the tunnel forwards to. Must match ``rllm.gateway.port`` in the run.
DEFAULT_PORT = 9090


@click.group("tunnel")
def tunnel():
    """Set up and run the public tunnel to the model gateway.

    Remote sandboxes (Daytona / Modal / Fireworks runtimes) reach the local
    gateway through this tunnel. A wildcard ngrok setup creates one tunnel per
    run; fixed domains use `rllm tunnel up` as before.
    """


@tunnel.command("setup")
def tunnel_setup():
    """Configure a stable tunnel backend (ngrok or cloudflared) for this machine."""
    from rllm.cli._ui import _select_from_menu, abort, console, fail
    from rllm.eval.config import load_tunnel_config, save_tunnel_config
    from rllm.gateway.tunnel import CloudflaredTunnel, NgrokTunnel

    existing = load_tunnel_config()
    backends = [
        ("ngrok", "ngrok — reserved domain or wildcard (wildcards need Pay-as-you-go)  [recommended]"),
        ("cloudflared", "cloudflared — free quick tunnel, zero setup, shared & rate-limited"),
    ]
    cursor = next((i for i, (b, _) in enumerate(backends) if b == existing.get("backend")), 0)
    idx = _select_from_menu("Tunnel backend", [d for _, d in backends], cursor)
    if idx is None:
        abort()
    backend = backends[idx][0]

    domain: str | None = None
    if backend == "ngrok":
        if not NgrokTunnel.is_available():
            fail(f"ngrok not found on PATH. {NgrokTunnel.install_hint}")
        token = click.prompt(
            "  ngrok authtoken (https://dashboard.ngrok.com/get-started/your-authtoken; blank if already configured)",
            default="",
            hide_input=True,
            show_default=False,
        ).strip()
        if token:
            try:
                subprocess.run(["ngrok", "config", "add-authtoken", token], check=True, capture_output=True, text=True)
                console.print("  [success]✓[/] ngrok authtoken saved.")
            except subprocess.CalledProcessError as e:
                fail(f"ngrok rejected the authtoken: {(e.stderr or '').strip() or e}")
        domain = (
            click.prompt(
                "  Reserved domain or wildcard (e.g. gateway.ngrok.app or *.rllm-team.ngrok.app)",
                default=existing.get("domain", ""),
                show_default=bool(existing.get("domain")),
            ).strip()
            or None
        )
        if (
            domain
            and domain.startswith("*.")
            and click.confirm(
                "  Reserve this wildcard with the ngrok API now?",
                default=domain != existing.get("domain"),
            )
        ):
            api_key = click.prompt(
                "  ngrok API key (used once, not saved; blank uses existing config; https://dashboard.ngrok.com/api)",
                default="",
                hide_input=True,
                show_default=False,
            ).strip()
            try:
                command = ["ngrok", "api", "reserved-domains", "create", "--domain", domain, "--description", "rLLM per-run gateway tunnels"]
                if api_key:
                    command.extend(["--api-key", api_key])
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                console.print(f"  [success]✓[/] Reserved [val]{domain}[/] with ngrok.")
            except subprocess.CalledProcessError as e:
                detail = (e.stderr or e.stdout or "").strip() or f"ngrok exited with status {e.returncode}"
                detail = detail.replace(api_key, "***") if api_key else detail
                fail(f"ngrok could not reserve {domain}: {detail}")
    else:
        if not CloudflaredTunnel.is_available():
            fail(f"cloudflared not found on PATH. {CloudflaredTunnel.install_hint}")
        console.print("  [muted]cloudflared quick tunnels are shared and rate-limited (HTTP 429); fine for smoke tests.[/]")

    port: int | None = None
    if not (backend == "ngrok" and domain and domain.startswith("*.")):
        port = click.prompt(
            "  Gateway port (must match rllm.gateway.port in training)",
            default=int(existing.get("port") or DEFAULT_PORT),
            type=int,
        )

    save_tunnel_config(backend, domain=domain, port=port)
    summary = backend + (f":{domain}" if domain else "")
    if port is None:
        console.print(f"\n  [success]✓ Tunnel configured:[/] [val]{summary}[/]")
        console.print("  Remote eval/train runs create a unique hostname and gateway port automatically.")
    else:
        console.print(f"\n  [success]✓ Tunnel configured:[/] [val]{summary}[/] [muted](gateway port {port})[/]")
        console.print("  Start it with [key]rllm tunnel up[/]; set its URL explicitly to reuse this fixed route.")


@tunnel.command("up")
@click.option("--backend", default=None, help="Override the configured backend (e.g. 'ngrok', 'ngrok:you.ngrok.dev', 'cloudflared').")
@click.option("--port", type=int, default=None, help="Gateway port to forward to (default: configured port or 9090).")
def tunnel_up(backend, port):
    """Start the tunnel as a background daemon and record its public URL."""
    from rllm.cli._ui import console, fail
    from rllm.eval.config import load_tunnel_config
    from rllm.gateway.tunnel import (
        TunnelStartError,
        create_tunnel,
        pid_alive,
        read_tunnel_state,
        spawn_detached,
        write_tunnel_state,
    )

    state = read_tunnel_state()
    if state and pid_alive(state.get("pid")):
        console.print(
            f"  [success]✓ Tunnel already running:[/] [val]{state.get('url')}[/] [muted](backend={state.get('backend')}, pid={state.get('pid')})[/]",
        )
        console.print("  Restart it with [key]rllm tunnel down && rllm tunnel up[/].")
        return

    cfg = load_tunnel_config()
    resolved_backend = backend or cfg.get("backend") or "cloudflared"
    # Fold a configured reserved domain into a bare "ngrok" spec.
    if resolved_backend == "ngrok" and cfg.get("domain"):
        resolved_backend = f"ngrok:{cfg['domain']}"
    resolved_port = port or cfg.get("port") or DEFAULT_PORT
    upstream = f"http://127.0.0.1:{resolved_port}"

    try:
        tnl = create_tunnel(resolved_backend, upstream)
    except ValueError as e:
        fail(str(e))

    console.print(f"  [cyan]…[/] Starting [val]{resolved_backend}[/] tunnel → [muted]{upstream}[/]")
    try:
        pid, url, log_path = spawn_detached(tnl)
    except TunnelStartError as e:
        fail(str(e))

    write_tunnel_state(backend=resolved_backend, url=url, pid=pid, upstream=upstream, log_path=log_path)
    console.print(f"  [success]✓ Tunnel up:[/] [val]{url}[/] [muted](pid {pid})[/]")
    console.print(f"  [label]logs[/] {log_path}")
        console.print(f"  To reuse it, set [key]RLLM_GATEWAY_TUNNEL[/] to this URL and [key]rllm.gateway.port={resolved_port}[/].")
        console.print("  Stop it with [key]rllm tunnel down[/].")


@tunnel.command("status")
def tunnel_status():
    """Show the running tunnel (if any)."""
    from rllm.cli._ui import console
    from rllm.gateway.tunnel import clear_tunnel_state, pid_alive, read_tunnel_state

    state = read_tunnel_state()
    if not state:
        console.print("  [muted]No tunnel recorded. Run [key]rllm tunnel up[/].[/]")
        return
    if not pid_alive(state.get("pid")):
        console.print(f"  [error]✗ Recorded tunnel is not running[/] [muted](stale pid {state.get('pid')}); clearing state.[/]")
        clear_tunnel_state()
        return
    console.print(f"  [success]● running[/]  [val]{state.get('url')}[/]")
    console.print(
        f"  [label]backend[/] {state.get('backend')}   [label]pid[/] {state.get('pid')}   [label]upstream[/] {state.get('upstream')}",
    )


@tunnel.command("down")
def tunnel_down():
    """Stop the running tunnel daemon."""
    from rllm.cli._ui import console
    from rllm.gateway.tunnel import clear_tunnel_state, pid_alive, read_tunnel_state, terminate_pid

    state = read_tunnel_state()
    if not state:
        console.print("  [muted]No tunnel recorded.[/]")
        return
    pid = state.get("pid")
    if pid and pid_alive(pid):
        terminate_pid(pid)
        console.print(f"  [success]✓ Stopped tunnel[/] [muted](pid {pid}, {state.get('url')})[/]")
    else:
        console.print("  [muted]Tunnel was not running; clearing state.[/]")
    clear_tunnel_state()
