"""Public-URL tunneling for the rLLM gateway.

Remote sandboxes (Modal, Daytona, E2B, Runloop, …) run in a different
network from the trainer and can't reach ``http://127.0.0.1:9090``.
:class:`CloudflaredTunnel` runs ``cloudflared tunnel --url <gateway>``
as a subprocess, scrapes the assigned ``*.trycloudflare.com`` URL out
of stderr, and exposes it as :attr:`public_url`. Callers (typically
:class:`GatewayManager`) plumb that URL into the
:class:`AgentConfig.base_url` the harness sees, and the agent inside
the remote sandbox calls back to the gateway via the public hostname.

Cloudflared's quick tunnels are anonymous (no login required) and live
for the lifetime of the subprocess — :meth:`stop` terminates the
subprocess and tears the tunnel down.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Cloudflared prints the assigned URL on stderr inside an ASCII frame.
_TRYCF_URL_RE = re.compile(r"https?://[a-zA-Z0-9.-]+\.trycloudflare\.com")

# Hard ceiling on how long we'll wait for cloudflared to publish a URL
# before assuming the binary is misbehaving.
_DEFAULT_READY_TIMEOUT = 30.0

# How many times to retry the cloudflared spawn when the QuickTunnel
# allocator endpoint returns a transient 5xx. Cloudflare's free-tier
# QuickTunnel backend is intermittently flaky and recovers within
# seconds; without retry the trainer dies on a transient blip.
_DEFAULT_MAX_ATTEMPTS = 4


# Patterns in cloudflared stderr that indicate a transient cloudflare-
# side failure (vs. a permanent local misconfiguration).
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
        """Spawn cloudflared with retry on transient 5xx from Cloudflare.

        Returns the public ``https://*.trycloudflare.com`` URL on success.
        Retries the subprocess up to ``max_attempts`` times when
        cloudflared exits with a recognised transient error (5xx from
        the QuickTunnel allocator). Raises :class:`TunnelStartError` on
        permanent failure (binary missing, all retries exhausted, or
        timeout without any URL).
        """
        if not self.is_available():
            raise TunnelStartError("cloudflared binary not found on PATH. Install: brew install cloudflared")

        last_error: TunnelStartError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._start_once()
            except TunnelStartError as e:
                last_error = e
                tail = "\n".join(self._stderr_buffer[-20:])
                if attempt < self.max_attempts and _is_transient(tail):
                    backoff = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "cloudflared start attempt %d/%d failed (transient); retrying in %ds",
                        attempt,
                        self.max_attempts,
                        backoff,
                    )
                    time.sleep(backoff)
                    self._reset_state()
                    continue
                raise
        # Should be unreachable; the loop either returns or raises.
        assert last_error is not None
        raise last_error

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
            # If cloudflared exits before publishing, surface the failure now.
            if self._proc.poll() is not None:
                tail = "\n".join(self._stderr_buffer[-20:])
                raise TunnelStartError(f"cloudflared exited (code {self._proc.returncode}) before publishing a URL.\nstderr tail:\n{tail}")

        # Hit the timeout without a URL — kill cloudflared so we don't
        # leak the process.
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
