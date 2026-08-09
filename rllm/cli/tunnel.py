"""rLLM CLI: set up and run the public tunnel that lets remote sandboxes reach the gateway.

``rllm tunnel setup`` records a backend + credentials in ``~/.rllm/config.json``
(per-user, so nothing personal lands in shared train scripts). ``rllm tunnel up``
runs that backend as a background daemon and writes its live URL to a state file;
training auto-discovers it (see :func:`rllm.gateway.tunnel.resolve_auto_tunnel`),
so no ``rllm.gateway.tunnel=...`` is needed in the run command.
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
    gateway through this tunnel. Run `rllm tunnel setup` once, then
    `rllm tunnel up`; training auto-discovers the live URL with no per-run config.
    """


@tunnel.command("setup")
def tunnel_setup():
    """Configure a stable tunnel backend (ngrok, cloudflare, or cloudflared) for this machine."""
    import os

    from rllm.cli._ui import _select_from_menu, abort, console, fail
    from rllm.eval.config import load_tunnel_config, save_tunnel_config
    from rllm.gateway.tunnel import ENV_CF_TUNNEL_TOKEN, CloudflaredTunnel, CloudflareNamedTunnel, NgrokTunnel

    existing = load_tunnel_config()
    backends = [
        ("ngrok", "ngrok — stable reserved domain, needs a (free) account  [recommended]"),
        ("cloudflare", "cloudflare — named (production) tunnel on your own domain, needs a Cloudflare account"),
        ("cloudflared", "cloudflared — free quick tunnel, zero setup, shared & rate-limited"),
    ]
    cursor = next((i for i, (b, _) in enumerate(backends) if b == existing.get("backend")), 0)
    idx = _select_from_menu("Tunnel backend", [d for _, d in backends], cursor)
    if idx is None:
        abort()
    backend = backends[idx][0]

    domain: str | None = None
    name: str | None = None
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
                "  Reserved domain (e.g. you.ngrok.dev; blank for an ephemeral *.ngrok-free.app URL)",
                default=existing.get("domain", ""),
                show_default=bool(existing.get("domain")),
            ).strip()
            or None
        )
    elif backend == "cloudflare":
        if not CloudflareNamedTunnel.is_available():
            fail(f"cloudflared not found on PATH. {CloudflareNamedTunnel.install_hint}")
        domain = click.prompt(
            "  Public hostname routed to the tunnel (e.g. rllm.example.com)",
            default=existing.get("domain", ""),
            show_default=bool(existing.get("domain")),
        ).strip()
        if not domain:
            fail("A public hostname is required for a Cloudflare named tunnel.")
        name = (
            click.prompt(
                "  Locally-managed tunnel name from `cloudflared tunnel create` (blank if dashboard-managed via a token)",
                default=existing.get("name", ""),
                show_default=bool(existing.get("name")),
            ).strip()
            or None
        )
        if name:
            console.print(f"  [muted]Requires `cloudflared tunnel login`, `cloudflared tunnel create {name}` and `cloudflared tunnel route dns {name} {domain}` once.[/]")
        else:
            console.print(f"  [muted]Dashboard-managed: export the connector token as TUNNEL_TOKEN (or {ENV_CF_TUNNEL_TOKEN}) before `rllm tunnel up`,[/]")
            console.print("  [muted]and point the tunnel's public hostname at http://localhost:<gateway port> in Zero Trust → Networks → Tunnels.[/]")
            if not (os.getenv("TUNNEL_TOKEN") or os.getenv(ENV_CF_TUNNEL_TOKEN)):
                console.print("  [yellow]![/] [muted]No TUNNEL_TOKEN in this shell yet.[/]")
    else:
        if not CloudflaredTunnel.is_available():
            fail(f"cloudflared not found on PATH. {CloudflaredTunnel.install_hint}")
        console.print("  [muted]cloudflared quick tunnels are shared and rate-limited (HTTP 429); fine for smoke tests.[/]")

    port = click.prompt(
        "  Gateway port (must match rllm.gateway.port in training)",
        default=int(existing.get("port") or DEFAULT_PORT),
        type=int,
    )

    save_tunnel_config(backend, domain=domain, name=name, port=int(port))
    summary = backend + (f":{domain}" if domain else "") + (f"@{name}" if name else "")
    console.print(f"\n  [success]✓ Tunnel configured:[/] [val]{summary}[/] [muted](gateway port {port})[/]")
    console.print("  Start it with [key]rllm tunnel up[/]; training picks up the URL automatically.")


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
    # Fold configured details into a bare backend spec.
    if resolved_backend == "ngrok" and cfg.get("domain"):
        resolved_backend = f"ngrok:{cfg['domain']}"
    elif resolved_backend == "cloudflare" and cfg.get("domain"):
        resolved_backend = f"cloudflare:{cfg['domain']}" + (f"@{cfg['name']}" if cfg.get("name") else "")
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
    console.print("  Training runs forward through this automatically. Stop it with [key]rllm tunnel down[/].")


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
