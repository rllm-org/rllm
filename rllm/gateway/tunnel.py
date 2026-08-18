"""Public-URL tunneling for the rLLM gateway.

:class:`CloudflaredTunnel` runs ``cloudflared tunnel --url <gateway>``
and exposes the assigned ``*.trycloudflare.com`` URL as
:attr:`public_url`. :class:`GatewayManager` plumbs that URL into
:class:`AgentConfig.base_url` so agents in remote sandboxes can reach
the gateway.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time

from rich.console import Console

from rllm.env import env_float

logger = logging.getLogger(__name__)

_status = Console()

# Sandbox backends that share network with the gateway host.
LOCAL_SANDBOX_BACKENDS: frozenset[str] = frozenset({"docker", "local", "apple-container"})


def is_local_sandbox_backend(name: str | None) -> bool:
    """True for backends that share network with the gateway host (``None`` treated as local)."""
    if not name:
        return True
    return name.lower() in LOCAL_SANDBOX_BACKENDS


def parse_tunnel(value: str | None) -> tuple[str | None, str | None]:
    """Split ``rllm.gateway.tunnel`` into ``(public_url, backend)``.

    URLs (``http(s)://...``) pass through as ``public_url``; anything
    else is treated as a backend name to spawn (e.g. ``"cloudflared"``).
    """
    if not value:
        return None, None
    if value.startswith(("http://", "https://")):
        return value, None
    return None, value.lower()


_TRYCF_URL_RE = re.compile(r"https?://[a-zA-Z0-9.-]+\.trycloudflare\.com")
_DEFAULT_READY_TIMEOUT = env_float("RLLM_TUNNEL_READY_TIMEOUT_S", 30.0)  # set env var: export RLLM_TUNNEL_READY_TIMEOUT_S=xxx
_DEFAULT_REACHABLE_TIMEOUT = env_float("RLLM_TUNNEL_REACHABLE_TIMEOUT_S", 120.0)  # set env var: export RLLM_TUNNEL_REACHABLE_TIMEOUT_S=xxx
# Retry transient Cloudflare QuickTunnel allocator 5xx blips.
_DEFAULT_MAX_ATTEMPTS = 4


_TRANSIENT_PATTERNS = (
    re.compile(r"500 Internal Server Error"),
    re.compile(r"502 Bad Gateway"),
    re.compile(r"503 Service Unavailable"),
    re.compile(r"504 Gateway Timeout"),
    re.compile(r"failed to unmarshal quick Tunnel"),
    re.compile(r"failed to request quick Tunnel"),
)


class TunnelStartError(RuntimeError):
    """Raised when the tunnel subprocess can't be brought up."""


class CloudflaredTunnel:
    """Run ``cloudflared tunnel --url <upstream>`` and surface the public URL."""

    def __init__(
        self,
        upstream_url: str,
        *,
        ready_timeout: float = _DEFAULT_READY_TIMEOUT,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.upstream_url = upstream_url
        self.ready_timeout = ready_timeout
        self.max_attempts = max_attempts
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._public_url: str | None = None
        self._url_event = threading.Event()
        self._stderr_buffer: list[str] = []

    @property
    def public_url(self) -> str | None:
        return self._public_url

    @staticmethod
    def is_available() -> bool:
        """True if a usable ``cloudflared`` binary is on PATH."""
        return shutil.which("cloudflared") is not None

    def start(self) -> str:
        """Spawn cloudflared with retry on transient 5xx; return the public URL or raise :class:`TunnelStartError`."""
        if not self.is_available():
            _status.print("[bold red]✗[/] cloudflared not found on PATH. Install: [bold]brew install cloudflared[/]")
            raise TunnelStartError("cloudflared binary not found on PATH. Install: brew install cloudflared")

        _status.print(f"  [cyan]…[/] Starting cloudflared tunnel for [bold]{self.upstream_url}[/]")
        last_error: TunnelStartError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                url = self._start_once()
                self._wait_reachable(url)
                _status.print(f"  [bold green]✓[/] Tunnel ready: [bold]{url}[/]")
                return url
            except TunnelStartError as e:
                last_error = e
                tail = "\n".join(self._stderr_buffer[-20:])
                if attempt < self.max_attempts and _is_transient(tail):
                    backoff = min(2 ** (attempt - 1), 8)
                    _status.print(
                        f"  [yellow]![/] cloudflared transient failure (attempt {attempt}/{self.max_attempts}); retrying in {backoff}s",
                    )
                    logger.warning(
                        "cloudflared start attempt %d/%d failed (transient); retrying in %ds",
                        attempt,
                        self.max_attempts,
                        backoff,
                    )
                    time.sleep(backoff)
                    self._reset_state()
                    continue
                _status.print(f"  [bold red]✗[/] cloudflared failed after {attempt} attempt(s): {e}")
                raise
        assert last_error is not None
        raise last_error

    def _wait_reachable(self, url: str, timeout: float = _DEFAULT_REACHABLE_TIMEOUT) -> None:
        """Block until the public URL answers over HTTP, or warn and continue.

        cloudflared printing the URL only means the hostname was *assigned* —
        not that this host can resolve it or that the edge has connected to
        the origin. Any response with status < 500 (404 included) proves
        DNS + edge + origin are live. Timeout is warn-not-raise: an
        in-sandbox consumer may still resolve the hostname through its own
        (cloud) DNS even when this host's resolver lags.
        """
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        while True:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as resp:
                    if resp.status < 500:
                        return
            except urllib.error.HTTPError as e:
                if e.code < 500:  # 4xx: resolvable and the origin answered
                    return
            except Exception:  # noqa: S110 — DNS/connect failures: keep polling
                pass
            if time.monotonic() > deadline:
                logger.warning("tunnel %s not reachable from this host after %.0fs — continuing (in-sandbox DNS may still resolve it)", url, timeout)
                return
            time.sleep(2)

    def _reset_state(self) -> None:
        """Clear per-attempt state before a retry."""
        self._proc = None
        self._reader = None
        self._public_url = None
        self._url_event = threading.Event()
        self._stderr_buffer = []

    def _start_once(self) -> str:
        cmd = ["cloudflared", "tunnel", "--no-autoupdate", "--url", self.upstream_url]
        logger.info("Starting cloudflared tunnel: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._scan_stderr, daemon=True)
        self._reader.start()

        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self._url_event.wait(timeout=0.5):
                assert self._public_url is not None
                logger.info("Cloudflared tunnel ready: %s -> %s", self._public_url, self.upstream_url)
                return self._public_url
            if self._proc.poll() is not None:
                tail = "\n".join(self._stderr_buffer[-20:])
                raise TunnelStartError(f"cloudflared exited (code {self._proc.returncode}) before publishing a URL.\nstderr tail:\n{tail}")

        self.stop()
        tail = "\n".join(self._stderr_buffer[-20:])
        raise TunnelStartError(f"cloudflared did not publish a URL within {self.ready_timeout}s.\nstderr tail:\n{tail}")

    def _scan_stderr(self) -> None:
        """Background thread: tee stderr into the buffer and pluck the URL."""
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                line = line.rstrip()
                self._stderr_buffer.append(line)
                if self._public_url is None:
                    m = _TRYCF_URL_RE.search(line)
                    if m:
                        self._public_url = m.group(0)
                        self._url_event.set()
        except Exception:
            logger.exception("cloudflared stderr reader crashed")

    def stop(self) -> None:
        """Terminate the cloudflared subprocess (if running)."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:
                logger.exception("Error terminating cloudflared")
        self._proc = None
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2)
        self._reader = None
        logger.info("Cloudflared tunnel stopped")


def _is_transient(stderr_tail: str) -> bool:
    """Whether a cloudflared stderr tail looks like a Cloudflare-side blip."""
    return any(p.search(stderr_tail) for p in _TRANSIENT_PATTERNS)
