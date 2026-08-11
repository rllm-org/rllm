"""Public-URL tunneling for the rLLM gateway.

Tunnels expose the local gateway at a public URL so agents in *remote*
sandboxes (Daytona, Modal, Fireworks-driven runtimes) can reach it. Three
backends ship:

* :class:`CloudflaredTunnel` — ``cloudflared tunnel --url`` →
  ``*.trycloudflare.com``. Zero-setup, but a shared, rate-limited (HTTP 429)
  quick tunnel; fine for smoke tests, not for high-concurrency training.
* :class:`CloudflareNamedTunnel` — ``cloudflared tunnel run`` against a
  *named* (production) tunnel tied to a Cloudflare account, serving a stable
  hostname on your own domain with none of the quick-tunnel limits.
* :class:`NgrokTunnel` — ``ngrok http`` → ``*.ngrok-free.app`` or your own
  reserved domain. Needs a one-time ``rllm tunnel setup`` (authtoken) but is
  stable across restarts.

:class:`GatewayManager` plumbs the chosen tunnel's :attr:`public_url` into
:class:`AgentConfig.base_url`. It builds a tunnel from the
``rllm.gateway.tunnel`` config via :func:`create_tunnel`.

The ``rllm tunnel`` CLI runs a backend as a *detached daemon*
(:func:`spawn_detached`) whose live URL is written to a state file
(:func:`write_tunnel_state`). Training then auto-discovers that URL via
:func:`resolve_auto_tunnel` with no per-run config — set
``rllm.gateway.tunnel`` explicitly only to override.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time

from rich.console import Console

from rllm.env import env_float
from rllm.paths import rllm_path

logger = logging.getLogger(__name__)

_status = Console()

# Sandbox backends that share network with the gateway host.
LOCAL_SANDBOX_BACKENDS: frozenset[str] = frozenset({"docker", "local", "apple-container"})

# Environment override consulted before the daemon state file (see
# :func:`resolve_auto_tunnel`). Either a backend name ("cloudflared",
# "ngrok", "ngrok:<domain>", "cloudflare:<hostname>[@<tunnel-name>]") or an
# explicit http(s):// URL.
ENV_TUNNEL = "RLLM_GATEWAY_TUNNEL"

# Connector token for a dashboard-managed Cloudflare named tunnel.
# ``TUNNEL_TOKEN`` is cloudflared's own env var; the rLLM-prefixed alias
# exists so the token can live next to other RLLM_* settings.
ENV_CF_TUNNEL_TOKEN = "CLOUDFLARE_TUNNEL_TOKEN"


def is_local_sandbox_backend(name: str | None) -> bool:
    """True for backends that share network with the gateway host (``None`` treated as local)."""
    if not name:
        return True
    return name.lower() in LOCAL_SANDBOX_BACKENDS


def parse_tunnel(value: str | None) -> tuple[str | None, str | None]:
    """Split ``rllm.gateway.tunnel`` into ``(public_url, backend_spec)``.

    URLs (``http(s)://...``) pass through as ``public_url``; anything else is
    treated as a backend spec to spawn (e.g. ``"cloudflared"``, ``"ngrok"``,
    ``"ngrok:rllm.ngrok.dev"``, ``"cloudflare:rllm.example.com@my-tunnel"``) —
    see :func:`create_tunnel`. The spec is passed through verbatim (Cloudflare
    tunnel names are case-sensitive); :func:`create_tunnel` normalizes the
    backend-name part itself.
    """
    if not value:
        return None, None
    if value.startswith(("http://", "https://")):
        return value, None
    return None, value


_TRYCF_URL_RE = re.compile(r"https?://[a-zA-Z0-9.-]+\.trycloudflare\.com")
# ngrok --log-format=json emits one record per line; the "started tunnel"
# record carries the public URL as a "url":"https://..." field.
_NGROK_URL_RE = re.compile(r'"url":"(https?://[^"]+)"')
_DEFAULT_READY_TIMEOUT = env_float("RLLM_TUNNEL_READY_TIMEOUT_S", 30.0)  # set env var: export RLLM_TUNNEL_READY_TIMEOUT_S=xxx
_DEFAULT_REACHABLE_TIMEOUT = env_float("RLLM_TUNNEL_REACHABLE_TIMEOUT_S", 120.0)  # set env var: export RLLM_TUNNEL_REACHABLE_TIMEOUT_S=xxx
# Retry transient Cloudflare QuickTunnel allocator 5xx blips.
_DEFAULT_MAX_ATTEMPTS = 4


_CF_TRANSIENT_PATTERNS = (
    re.compile(r"500 Internal Server Error"),
    re.compile(r"502 Bad Gateway"),
    re.compile(r"503 Service Unavailable"),
    re.compile(r"504 Gateway Timeout"),
    re.compile(r"failed to unmarshal quick Tunnel"),
    re.compile(r"failed to request quick Tunnel"),
)


class TunnelStartError(RuntimeError):
    """Raised when the tunnel subprocess can't be brought up."""


class _Tunnel:
    """Base for ``<binary> ...`` tunnels that publish a public URL on a log stream.

    Subclasses set :attr:`name`, :attr:`binary`, :attr:`install_hint`,
    :attr:`log_stream` (which pipe carries the URL) and implement
    :meth:`_command` / :meth:`_extract_url`. They may override
    :meth:`_is_transient` (retry-worthy blips) and :meth:`_classify_fatal`
    (turn a known error into an actionable hint).
    """

    name: str = "tunnel"
    binary: str = ""
    install_hint: str = ""
    log_stream: str = "stderr"  # "stderr" (cloudflared) or "stdout" (ngrok)

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
        self._log_buffer: list[str] = []

    @property
    def public_url(self) -> str | None:
        return self._public_url

    @classmethod
    def is_available(cls) -> bool:
        """True if a usable ``<binary>`` is on PATH."""
        return shutil.which(cls.binary) is not None

    # -- subclass hooks ------------------------------------------------------

    def _command(self) -> list[str]:
        raise NotImplementedError

    def _popen_env(self) -> dict[str, str] | None:
        """Environment for the tunnel subprocess (``None`` = inherit the caller's)."""
        return None

    def _extract_url(self, line: str) -> str | None:
        """Return the public URL if this log line announces it, else ``None``."""
        raise NotImplementedError

    def _is_transient(self, log_tail: str) -> bool:
        """Whether a failure looks like a retry-worthy upstream blip."""
        return False

    def _classify_fatal(self, log_tail: str) -> str | None:
        """Map a known-fatal failure to an actionable hint (no retry), else ``None``."""
        return None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> str:
        """Spawn the tunnel (retrying transient blips); return the public URL or raise :class:`TunnelStartError`."""
        if not self.is_available():
            _status.print(f"[bold red]✗[/] {self.binary} not found on PATH. {self.install_hint}")
            raise TunnelStartError(f"{self.binary} binary not found on PATH. {self.install_hint}")

        _status.print(f"  [cyan]…[/] Starting {self.name} tunnel for [bold]{self.upstream_url}[/]")
        last_error: TunnelStartError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                url = self._start_once()
                self._wait_reachable(url)
                _status.print(f"  [bold green]✓[/] Tunnel ready: [bold]{url}[/]")
                return url
            except TunnelStartError as e:
                last_error = e
                tail = "\n".join(self._log_buffer[-20:])
                hint = self._classify_fatal(tail)
                if hint:
                    _status.print(f"  [bold red]✗[/] {self.name}: {hint}")
                    raise TunnelStartError(hint) from e
                if attempt < self.max_attempts and self._is_transient(tail):
                    backoff = min(2 ** (attempt - 1), 8)
                    _status.print(
                        f"  [yellow]![/] {self.name} transient failure (attempt {attempt}/{self.max_attempts}); retrying in {backoff}s",
                    )
                    logger.warning(
                        "%s start attempt %d/%d failed (transient); retrying in %ds",
                        self.name,
                        attempt,
                        self.max_attempts,
                        backoff,
                    )
                    time.sleep(backoff)
                    self._reset_state()
                    continue
                _status.print(f"  [bold red]✗[/] {self.name} failed after {attempt} attempt(s): {e}")
                raise
        assert last_error is not None
        raise last_error

    def _wait_reachable(self, url: str, timeout: float = _DEFAULT_REACHABLE_TIMEOUT) -> None:
        """Block until the public URL answers over HTTP, or warn and continue.

        A backend printing the URL only means the hostname was *assigned* — not
        that this host can resolve it or that the edge has connected to the
        origin. Any response with status < 500 (404 included) proves
        DNS + edge + origin are live. Timeout is warn-not-raise: an in-sandbox
        consumer may still resolve the hostname through its own (cloud) DNS even
        when this host's resolver lags.
        """
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        # Identify as a real client: Cloudflare bot protection 403s urllib's
        # default "Python-urllib/x.y" UA at the edge, which would make this
        # probe "pass" without ever traversing the tunnel.
        headers = {"User-Agent": "rllm-tunnel-healthcheck"}
        while True:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="GET", headers=headers), timeout=5) as resp:
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
        self._log_buffer = []

    def _start_once(self) -> str:
        cmd = self._command()
        logger.info("Starting %s tunnel: %s", self.name, " ".join(cmd))

        if self.log_stream == "stderr":
            stdout, stderr = subprocess.DEVNULL, subprocess.PIPE
        else:
            stdout, stderr = subprocess.PIPE, subprocess.STDOUT

        self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, bufsize=1, env=self._popen_env())
        self._reader = threading.Thread(target=self._scan_stream, daemon=True)
        self._reader.start()

        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self._url_event.wait(timeout=0.5):
                assert self._public_url is not None
                logger.info("%s tunnel ready: %s -> %s", self.name, self._public_url, self.upstream_url)
                return self._public_url
            if self._proc.poll() is not None:
                tail = "\n".join(self._log_buffer[-20:])
                raise TunnelStartError(f"{self.binary} exited (code {self._proc.returncode}) before publishing a URL.\nlog tail:\n{tail}")

        self.stop()
        tail = "\n".join(self._log_buffer[-20:])
        raise TunnelStartError(f"{self.binary} did not publish a URL within {self.ready_timeout}s.\nlog tail:\n{tail}")

    def _scan_stream(self) -> None:
        """Background thread: tee the log stream into the buffer and pluck the URL."""
        assert self._proc is not None
        stream = self._proc.stderr if self.log_stream == "stderr" else self._proc.stdout
        assert stream is not None
        try:
            for line in stream:
                line = line.rstrip()
                self._log_buffer.append(line)
                if self._public_url is None:
                    url = self._extract_url(line)
                    if url:
                        self._public_url = url
                        self._url_event.set()
        except Exception:
            logger.exception("%s log reader crashed", self.name)

    def stop(self) -> None:
        """Terminate the tunnel subprocess (if running)."""
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
                logger.exception("Error terminating %s", self.binary)
        self._proc = None
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=2)
        self._reader = None
        logger.info("%s tunnel stopped", self.name)


class CloudflaredTunnel(_Tunnel):
    """Run ``cloudflared tunnel --url <upstream>`` and surface the ``*.trycloudflare.com`` URL."""

    name = "cloudflared"
    binary = "cloudflared"
    install_hint = "Install: brew install cloudflared"
    log_stream = "stderr"

    def _command(self) -> list[str]:
        return ["cloudflared", "tunnel", "--no-autoupdate", "--url", self.upstream_url]

    def _extract_url(self, line: str) -> str | None:
        m = _TRYCF_URL_RE.search(line)
        return m.group(0) if m else None

    def _is_transient(self, log_tail: str) -> bool:
        return any(p.search(log_tail) for p in _CF_TRANSIENT_PATTERNS)


class NgrokTunnel(_Tunnel):
    """Run ``ngrok http <upstream>`` and surface the assigned ngrok URL.

    Pass ``domain`` to pin a reserved domain (e.g. ``rllm.ngrok.dev``);
    otherwise ngrok assigns an ephemeral ``*.ngrok-free.app`` URL. ngrok needs
    an authtoken — run ``rllm tunnel setup`` or ``ngrok config add-authtoken``.
    """

    name = "ngrok"
    binary = "ngrok"
    install_hint = "Install: https://ngrok.com/download (or `brew install ngrok`)"
    log_stream = "stdout"

    def __init__(self, upstream_url: str, *, domain: str | None = None, **kwargs) -> None:
        super().__init__(upstream_url, **kwargs)
        self.domain = domain.strip() if domain else None

    def _command(self) -> list[str]:
        # ngrok http takes a local addr/port; strip the scheme so we pass
        # "127.0.0.1:9090" rather than a full URL.
        target = self.upstream_url.split("://", 1)[-1]
        cmd = ["ngrok", "http", target, "--log", "stdout", "--log-format", "json"]
        if self.domain:
            url = self.domain if "://" in self.domain else f"https://{self.domain}"
            cmd += ["--url", url]
        return cmd

    def _extract_url(self, line: str) -> str | None:
        m = _NGROK_URL_RE.search(line)
        if not m:
            return None
        url = m.group(1)
        # The JSON log also mentions the local web-inspector URL; skip it.
        if "127.0.0.1" in url or "localhost" in url:
            return None
        return url

    def _classify_fatal(self, log_tail: str) -> str | None:
        low = log_tail.lower()
        if "err_ngrok_4018" in low or ("authtoken" in low and ("required" in low or "authenticate" in low)):
            return "ngrok needs an authtoken. Run `rllm tunnel setup` (or `ngrok config add-authtoken <token>`)."
        if "err_ngrok_108" in low or "simultaneous ngrok agent session" in low:
            return "An ngrok agent is already running for this account. Stop it with `rllm tunnel down` (or kill the other `ngrok` process)."
        if "err_ngrok_3200" in low or ("domain" in low and "not found" in low):
            return f"ngrok could not bind the reserved domain {self.domain!r}. Check it exists on your account (https://dashboard.ngrok.com/domains)."
        return None


class CloudflareNamedTunnel(_Tunnel):
    """Run a *named* (production) Cloudflare tunnel serving a stable hostname.

    Unlike :class:`CloudflaredTunnel` quick tunnels, named tunnels are tied to
    a Cloudflare account and serve a hostname on your own domain, with no
    shared-infra rate limits or 120s origin read timeout. Two modes:

    * **Dashboard-managed (token)** — create the tunnel in Zero Trust
      (Networks → Tunnels), point its public hostname at
      ``http://localhost:<gateway port>``, and export the connector token as
      ``TUNNEL_TOKEN`` (or ``CLOUDFLARE_TUNNEL_TOKEN``). Spec:
      ``cloudflare:<hostname>``. Ingress lives in the dashboard, so the
      hostname → port mapping there must match the gateway port.
    * **Locally-managed (name)** — after ``cloudflared tunnel login``,
      ``cloudflared tunnel create <name>`` and
      ``cloudflared tunnel route dns <name> <hostname>``, we run
      ``cloudflared tunnel run --url <upstream> <name>`` so ingress follows the
      gateway port automatically. Spec: ``cloudflare:<hostname>@<name>``.

    Named tunnels never log their public URL (the hostname is DNS-routed), so
    the URL is derived from ``hostname`` once cloudflared registers its first
    connection to the Cloudflare edge.
    """

    name = "cloudflare"
    binary = "cloudflared"
    install_hint = "Install: brew install cloudflared"
    log_stream = "stderr"

    _REGISTERED_RE = re.compile(r"Registered tunnel connection")

    def __init__(self, upstream_url: str, *, hostname: str, tunnel_ref: str | None = None, **kwargs) -> None:
        super().__init__(upstream_url, **kwargs)
        self.hostname = hostname.strip().split("://", 1)[-1].rstrip("/")
        self.tunnel_ref = tunnel_ref.strip() if tunnel_ref else None

    @staticmethod
    def _token() -> str | None:
        return os.getenv("TUNNEL_TOKEN") or os.getenv(ENV_CF_TUNNEL_TOKEN)

    def _command(self) -> list[str]:
        if self.tunnel_ref:
            return ["cloudflared", "tunnel", "run", "--url", self.upstream_url, self.tunnel_ref]
        if not self._token():
            raise TunnelStartError(
                f"Cloudflare named tunnel for {self.hostname!r} needs credentials: export the connector "
                f"token as TUNNEL_TOKEN or {ENV_CF_TUNNEL_TOKEN} (Zero Trust → Networks → Tunnels), or use a "
                "locally-managed tunnel via 'cloudflare:<hostname>@<tunnel-name>'.",
            )
        # Dashboard-managed: ingress (hostname → local port) lives in Zero
        # Trust, so no --url here; the token travels via the TUNNEL_TOKEN env
        # var (see _popen_env) to keep it out of `ps` output.
        return ["cloudflared", "tunnel", "run"]

    def _popen_env(self) -> dict[str, str] | None:
        if self.tunnel_ref:
            return None
        return {**os.environ, "TUNNEL_TOKEN": self._token() or ""}

    def _extract_url(self, line: str) -> str | None:
        # Named tunnels never announce a public URL; the first edge
        # registration means "serving", and the hostname is known up front.
        if self._REGISTERED_RE.search(line):
            return f"https://{self.hostname}"
        return None

    def _is_transient(self, log_tail: str) -> bool:
        return any(p.search(log_tail) for p in _CF_TRANSIENT_PATTERNS)

    def _classify_fatal(self, log_tail: str) -> str | None:
        low = log_tail.lower()
        if "token is not valid" in low or "invalid tunnel token" in low:
            return "Cloudflare rejected the tunnel token. Re-copy it from Zero Trust → Networks → Tunnels → your tunnel, then re-export TUNNEL_TOKEN."
        if "cert.pem" in low or "origincert" in low:
            return f"cloudflared has no origin certificate for locally-managed tunnels. Run `cloudflared tunnel login`, then `cloudflared tunnel create {self.tunnel_ref or '<name>'}`."
        if "credentials file" in low or ("tunnel" in low and "not found" in low):
            return f"cloudflared could not load tunnel {self.tunnel_ref!r}. Check `cloudflared tunnel list` and the credentials JSON in ~/.cloudflared/."
        return None


# Registry of spawnable backends keyed by the name in ``rllm.gateway.tunnel``.
_BACKENDS: dict[str, type[_Tunnel]] = {
    "cloudflared": CloudflaredTunnel,
    "cloudflare": CloudflareNamedTunnel,
    "ngrok": NgrokTunnel,
}


def create_tunnel(spec: str, upstream_url: str) -> _Tunnel:
    """Build a tunnel from a backend spec.

    Accepted specs: ``"cloudflared"`` (quick tunnel), ``"ngrok"``,
    ``"ngrok:<domain>"``, ``"cloudflare:<hostname>"`` (dashboard-managed named
    tunnel, token from ``$TUNNEL_TOKEN``), or
    ``"cloudflare:<hostname>@<tunnel-name>"`` (locally-managed named tunnel).

    Raises :class:`ValueError` for an unknown backend or a malformed spec.
    """
    name, _, opt = spec.partition(":")
    name = name.strip().lower()
    cls = _BACKENDS.get(name)
    if cls is None:
        supported = "', '".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unsupported gateway tunnel backend: {spec!r}. Supported: '{supported}', 'ngrok:<domain>', 'cloudflare:<hostname>[@<tunnel-name>]', or an http(s):// URL.",
        )
    if cls is NgrokTunnel:
        return NgrokTunnel(upstream_url, domain=opt or None)
    if cls is CloudflareNamedTunnel:
        hostname, _, tunnel_ref = opt.partition("@")
        if not hostname.strip():
            raise ValueError(
                f"Cloudflare named tunnels need a public hostname: use 'cloudflare:<hostname>' or 'cloudflare:<hostname>@<tunnel-name>' (got {spec!r}).",
            )
        return CloudflareNamedTunnel(upstream_url, hostname=hostname, tunnel_ref=tunnel_ref or None)
    return cls(upstream_url)


# ---------------------------------------------------------------------------
# Detached daemon + state file (used by the ``rllm tunnel`` CLI)
#
# ``rllm tunnel up`` runs a backend as a session-leader process that outlives
# the CLI, captures its public URL from a log file, and records it so training
# runs can pick it up without per-run config.
# ---------------------------------------------------------------------------


def daemon_log_path(backend_name: str) -> str:
    return rllm_path(f"tunnel-{backend_name}.log")


def spawn_detached(tunnel: _Tunnel, *, ready_timeout: float = _DEFAULT_READY_TIMEOUT) -> tuple[int, str, str]:
    """Spawn ``tunnel`` as a detached daemon; return ``(pid, public_url, log_path)``.

    The process is started in its own session so it survives the caller
    exiting. Raises :class:`TunnelStartError` if no URL is published in time.
    """
    if not tunnel.is_available():
        raise TunnelStartError(f"{tunnel.binary} binary not found on PATH. {tunnel.install_hint}")

    log_path = daemon_log_path(tunnel.name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")  # noqa: SIM115 — handed to the child; closed below
    try:
        proc = subprocess.Popen(
            tunnel._command(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=tunnel._popen_env(),
        )
    finally:
        log.close()

    url = _tail_log_for_url(log_path, tunnel, proc, ready_timeout)
    if url is None:
        tail = _read_log_tail(log_path)
        hint = tunnel._classify_fatal(tail)
        terminate_pid(proc.pid)
        if hint:
            raise TunnelStartError(hint)
        if proc.poll() is not None:
            raise TunnelStartError(f"{tunnel.binary} exited (code {proc.returncode}) before publishing a URL.\nlog tail:\n{tail}")
        raise TunnelStartError(f"{tunnel.binary} did not publish a URL within {ready_timeout:.0f}s.\nlog tail:\n{tail}")
    return proc.pid, url, log_path


def _tail_log_for_url(log_path: str, tunnel: _Tunnel, proc: subprocess.Popen, timeout: float) -> str | None:
    deadline = time.monotonic() + timeout
    with open(log_path) as f:
        while time.monotonic() < deadline:
            line = f.readline()
            if line:
                url = tunnel._extract_url(line.rstrip())
                if url:
                    return url
                continue
            if proc.poll() is not None:
                # Drain anything written between the last readline and exit.
                for rest in f:
                    url = tunnel._extract_url(rest.rstrip())
                    if url:
                        return url
                return None
            time.sleep(0.2)
    return None


def _read_log_tail(log_path: str, n: int = 20) -> str:
    try:
        with open(log_path) as f:
            return "\n".join(f.read().splitlines()[-n:])
    except OSError:
        return ""


def terminate_pid(pid: int) -> None:
    """Terminate a detached tunnel by its process-group leader pid."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tunnel_state_path() -> str:
    return rllm_path("tunnel.json")


def write_tunnel_state(*, backend: str, url: str, pid: int, upstream: str, log_path: str | None = None) -> str:
    """Record the running daemon tunnel so training can auto-discover its URL."""
    path = tunnel_state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"backend": backend, "url": url, "pid": pid, "upstream": upstream, "log_path": log_path}, f, indent=2)
        f.write("\n")
    return path


def read_tunnel_state() -> dict | None:
    path = tunnel_state_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_tunnel_state() -> None:
    try:
        os.remove(tunnel_state_path())
    except FileNotFoundError:
        pass


def live_tunnel() -> dict | None:
    """State of a currently-running daemon tunnel, or ``None`` (clearing stale state).

    The dict is the state file as written by ``rllm tunnel up``: ``backend``,
    ``url`` (public), ``pid``, ``upstream`` (the local ``http://127.0.0.1:<port>``
    the daemon forwards to — the authoritative answer to "which port must the
    gateway bind"), ``log_path``.
    """
    state = read_tunnel_state()
    if not state:
        return None
    if pid_alive(state.get("pid")) and state.get("url"):
        return state
    clear_tunnel_state()
    return None


def live_tunnel_url() -> str | None:
    """URL of a currently-running daemon tunnel, or ``None`` (clearing stale state)."""
    state = live_tunnel()
    return state["url"] if state else None


def resolve_auto_tunnel() -> tuple[str, str | None]:
    """Decide ``rllm.gateway.tunnel`` when it is unset.

    Resolution order: ``$RLLM_GATEWAY_TUNNEL`` → a running ``rllm tunnel up``
    daemon → a free Cloudflare quick tunnel. Returns ``(value, warning)`` where
    ``warning`` is a ready-to-log message when falling back to the quick tunnel
    (and ``None`` otherwise).
    """
    env = os.getenv(ENV_TUNNEL)
    if env:
        return env, None
    live = live_tunnel_url()
    if live:
        return live, None

    try:
        from rllm.eval.config import load_tunnel_config

        cfg = load_tunnel_config()
    except Exception:
        cfg = {}
    if cfg.get("backend"):
        warning = (
            f"Tunnel backend {cfg['backend']!r} is configured but no tunnel is running — "
            "run `rllm tunnel up` for a stable tunnel. "
            "Falling back to a free Cloudflare quick tunnel (shared, rate-limited)."
        )
    else:
        warning = (
            "No gateway tunnel configured — falling back to a free Cloudflare quick tunnel "
            "(*.trycloudflare.com): shared infra, rate-limited (HTTP 429) and unsuitable for "
            "high-concurrency training. Run `rllm tunnel setup` then `rllm tunnel up` for a stable tunnel."
        )
    return "cloudflared", warning
