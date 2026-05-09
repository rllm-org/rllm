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


class TunnelStartError(RuntimeError):
    """Raised when the tunnel subprocess can't be brought up."""


class CloudflaredTunnel:
    """Run ``cloudflared tunnel --url <upstream>`` and surface the public URL."""

    def __init__(self, upstream_url: str, *, ready_timeout: float = _DEFAULT_READY_TIMEOUT) -> None:
        self.upstream_url = upstream_url
        self.ready_timeout = ready_timeout
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
        """Spawn cloudflared and block until the public URL is published.

        Returns the public ``https://*.trycloudflare.com`` URL on success.
        Raises :class:`TunnelStartError` if cloudflared exits before
        publishing a URL or if the URL doesn't appear within
        ``ready_timeout``.
        """
        if not self.is_available():
            raise TunnelStartError("cloudflared binary not found on PATH. Install: brew install cloudflared")

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
