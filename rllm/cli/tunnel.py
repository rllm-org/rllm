"""rLLM CLI: set up and run the public tunnel that lets remote sandboxes reach the gateway.

``rllm tunnel setup`` records a backend in ``~/.rllm/config.json`` (per-user,
so nothing personal lands in shared train scripts). A configured ngrok wildcard
creates an owned endpoint for each eval/train run. Legacy fixed domains and
cloudflared may instead run as a background daemon via ``rllm tunnel up``.
"""

from __future__ import annotations

import json
import secrets
import subprocess

import click

# Gateway port the tunnel forwards to. Must match ``rllm.gateway.port`` in the run.
DEFAULT_PORT = 9090


def _ngrok_reserved_domain_exists(domain: str) -> bool:
    """Return whether the authenticated ngrok account already owns ``domain``."""
    result = subprocess.run(
        [
            "ngrok",
            "api",
            "reserved-domains",
            "list",
            "--limit",
            "1",
            "--filter",
            f'obj.domain == "{domain}"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload_start = result.stdout.find("{")
    if payload_start < 0:
        raise ValueError("ngrok returned no JSON payload")
    payload = json.loads(result.stdout[payload_start:])
    return any(entry.get("domain") == domain for entry in payload.get("reserved_domains", []))


@click.group("tunnel")
def tunnel():
    """Set up and run the public tunnel to the model gateway.

    Remote sandboxes (Daytona / Modal / Fireworks runtimes) reach the local
    gateway through this tunnel. Run `rllm tunnel setup` once. An ngrok
    wildcard is allocated per eval/train run; fixed domains and cloudflared can
    still be run as a background daemon with `rllm tunnel up`.
    """


@tunnel.command("setup")
def tunnel_setup():
    """Configure a tunnel backend (ngrok or cloudflared) for this machine."""
    from rllm.cli._ui import _select_from_menu, abort, console, fail
    from rllm.eval.config import load_tunnel_config, save_tunnel_config
    from rllm.gateway.tunnel import CloudflaredTunnel, NgrokTunnel, is_ngrok_wildcard_domain

    existing = load_tunnel_config()
    backends = [
        ("ngrok", "ngrok — per-run wildcard (Pay-as-you-go) or fixed reserved domain  [recommended]"),
        ("cloudflared", "cloudflared — free quick tunnel, zero setup, shared & rate-limited"),
    ]
    cursor = next((i for i, (b, _) in enumerate(backends) if b == existing.get("backend")), 0)
    idx = _select_from_menu("Tunnel backend", [d for _, d in backends], cursor)
    if idx is None:
        abort()
    backend = backends[idx][0]

    domain: str | None = None
    per_run = False
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
                detail = (e.stderr or "").strip() or f"exit code {e.returncode}"
                fail(f"ngrok rejected the authtoken: {detail}")

        existing_domain = str(existing.get("domain") or "")
        domain_modes = [
            "Create a persistent wildcard with the ngrok API  [recommended]",
            "Use an existing wildcard",
            "Use a fixed reserved domain  [one run at a time]",
        ]
        mode_cursor = 1 if is_ngrok_wildcard_domain(existing_domain) else (2 if existing_domain else 0)
        mode = _select_from_menu("ngrok domain", domain_modes, mode_cursor)
        if mode is None:
            abort()

        if mode in (0, 1):
            per_run = True
            default_domain = existing_domain if is_ngrok_wildcard_domain(existing_domain) else f"*.rllm-{secrets.token_hex(4)}.ngrok.app"
            domain = click.prompt(
                "  Wildcard domain (for example *.rllm-team.ngrok.app)",
                default=default_domain,
            ).strip()
            if not is_ngrok_wildcard_domain(domain):
                fail("An ngrok wildcard must be a valid DNS wildcard such as '*.rllm-team.ngrok.app'.")

            if mode == 0:
                api_key = click.prompt(
                    "  ngrok API key (https://dashboard.ngrok.com/api-keys; blank if already configured)",
                    default="",
                    hide_input=True,
                    show_default=False,
                ).strip()
                if api_key:
                    try:
                        subprocess.run(["ngrok", "config", "add-api-key", api_key], check=True, capture_output=True, text=True)
                        console.print("  [success]✓[/] ngrok API key saved.")
                    except subprocess.CalledProcessError as e:
                        detail = (e.stderr or "").strip() or f"exit code {e.returncode}"
                        fail(f"ngrok rejected the API key: {detail}")
                try:
                    already_reserved = _ngrok_reserved_domain_exists(domain)
                except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
                    detail = ((e.stderr or e.stdout or "").strip() if isinstance(e, subprocess.CalledProcessError) else str(e)) or str(e)
                    fail(f"ngrok could not check {domain}: {detail}")
                if already_reserved:
                    console.print(f"  [success]✓[/] [val]{domain}[/] is already reserved on this ngrok account.")
                else:
                    try:
                        subprocess.run(
                            [
                                "ngrok",
                                "api",
                                "reserved-domains",
                                "create",
                                "--domain",
                                domain,
                                "--description",
                                "rLLM per-run gateway tunnels",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        console.print(f"  [success]✓[/] Reserved [val]{domain}[/] with ngrok.")
                    except subprocess.CalledProcessError as e:
                        # A second setup process may have won the race between
                        # our list and create calls. Verify ownership once more
                        # before treating the create failure as fatal.
                        try:
                            created_by_peer = _ngrok_reserved_domain_exists(domain)
                        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
                            created_by_peer = False
                        if created_by_peer:
                            console.print(f"  [success]✓[/] [val]{domain}[/] is already reserved on this ngrok account.")
                        else:
                            detail = (e.stderr or e.stdout or "").strip() or str(e)
                            fail(f"ngrok could not reserve {domain}: {detail}")
        else:
            domain = click.prompt(
                "  Fixed reserved domain (for example gateway.ngrok.app)",
                default=existing_domain,
                show_default=bool(existing_domain),
            ).strip()
            if not domain:
                fail("A fixed ngrok domain is required for this mode.")
            if "*" in domain:
                fail("A fixed ngrok domain cannot contain '*'; choose the wildcard mode for per-run endpoints.")
            console.print("  [muted]A fixed domain can point to only one gateway at a time; concurrent runs need a wildcard.[/]")
    else:
        if not CloudflaredTunnel.is_available():
            fail(f"cloudflared not found on PATH. {CloudflaredTunnel.install_hint}")
        console.print("  [muted]cloudflared quick tunnels are shared and rate-limited (HTTP 429); fine for smoke tests.[/]")

    port: int | None = None
    if not per_run:
        port = click.prompt(
            "  Gateway port (must match rllm.gateway.port in training)",
            default=int(existing.get("port") or DEFAULT_PORT),
            type=int,
        )

    save_tunnel_config(backend, domain=domain, port=port)
    summary = backend + (f":{domain}" if domain else "")
    if per_run:
        console.print(f"\n  [success]✓ Tunnel configured:[/] [val]{summary}[/]")
        console.print("  Each remote eval/train run now creates its own ngrok hostname and gateway port automatically.")
    else:
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
        is_ngrok_wildcard_spec,
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
    if is_ngrok_wildcard_spec(resolved_backend):
        fail("ngrok wildcards are created per eval/train run; run eval or train directly instead of `rllm tunnel up`.")
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
