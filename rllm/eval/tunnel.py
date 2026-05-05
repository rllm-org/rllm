"""Cloudflare-tunnel helper for exposing the local gateway to remote sandboxes.

Modal/Daytona/e2b sandboxes can't reach the eval host's loopback. This
module spawns ``cloudflared tunnel --url http://localhost:<port>``,
parses the printed ``trycloudflare.com`` URL from stdout, and hands
back the URL plus the running subprocess so the gateway manager can
terminate it on shutdown.

Anonymous tunnels (no Cloudflare account) are ephemeral: random
subdomain, ~24 h TTL, no inbound auth — every request that lands here
is publicly reachable. The gateway's ``inbound_auth_token`` is the
mandatory companion to this module; never spawn a tunnel without one.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time

logger = logging.getLogger(__name__)

# cloudflared writes the tunnel URL on stderr (which we merge into
# stdout) as e.g. ``Your quick Tunnel has been created!  ... https://abc-def.trycloudflare.com``.
_TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    """Cloudflared failed to come up cleanly."""


def start_cloudflared_tunnel(
    local_port: int,
    *,
    timeout: float = 30.0,
    binary: str = "cloudflared",
) -> tuple[str, subprocess.Popen]:
    """Spawn cloudflared and return ``(public_url, proc)`` once the URL appears.

    Raises :class:`TunnelError` with an actionable message if the binary
    is missing or doesn't surface a URL within *timeout*.
    """
    try:
        proc = subprocess.Popen(
            [binary, "tunnel", "--url", f"http://localhost:{local_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered so we don't wait on a pipe-full chunk
        )
    except FileNotFoundError as e:
        raise TunnelError(
            f"`{binary}` not found on PATH. Install it (macOS: `brew install cloudflared`; "
            "Linux: download from https://github.com/cloudflare/cloudflared/releases or your "
            "package manager) or pass --gateway-public-url to use your own tunnel/public IP."
        ) from e

    deadline = time.time() + timeout
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            # Pipe closed → process exited before we saw a URL.
            exit_code = proc.poll()
            if exit_code is not None:
                raise TunnelError(f"cloudflared exited with code {exit_code} before producing a URL")
            # Empty line, process still running — keep polling.
            continue
        match = _TUNNEL_URL_RE.search(line)
        if match:
            url = match.group(0)
            logger.info("cloudflared tunnel up at %s -> http://localhost:%d", url, local_port)
            return url, proc

    # Timed out — clean up before raising so we don't leak the proc.
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise TunnelError(f"cloudflared did not surface a public URL within {timeout:.0f}s")


def stop_tunnel(proc: subprocess.Popen | None, *, grace: float = 5.0) -> None:
    """Terminate a running cloudflared subprocess. Idempotent."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        logger.warning("cloudflared did not exit within %.1fs; killing", grace)
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
