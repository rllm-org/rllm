"""GatewayManager + EvalGatewayManager: rllm-model-gateway lifecycle.

Two classes share most of the lifecycle (uvicorn-on-thread or subprocess,
trace store, session API). They differ in how upstream workers are
registered and what request-body injection the middleware applies.

* :class:`GatewayManager` — training. Workers come from a verl/tinker
  rollout engine via :meth:`start(rollout_engine)`. Injects ``logprobs``
  and ``return_token_ids`` into request bodies (vLLM needs both for the
  loss math downstream).

* :class:`EvalGatewayManager(GatewayManager)` — eval. Wraps a static
  upstream URL (vLLM endpoint, LiteLLM proxy, OpenAI-compatible server).
  Disables vLLM-specific param injection because external providers 400
  on ``return_token_ids``. Constructed with
  ``EvalGatewayManager(upstream_url, model)`` and started with
  ``.start()`` (no rollout engine).

Modes:

- 'process': subprocess via ``rllm-model-gateway`` CLI (for verl / distributed)
- 'thread': background thread via ``create_app`` + uvicorn (for tinker /
  single-machine / eval)

For Tinker backends, an in-process handler is injected into the gateway
(via ``local_handler``), avoiding the need for a separate HTTP backend server.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from rllm_model_gateway.client import AsyncGatewayClient, GatewayClient
from rllm_model_gateway.models import TraceRecord

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.experimental.rollout import RolloutEngine, VerlEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0
_TRACE_API_TIMEOUT = 600.0


def _find_free_port() -> int:
    """Ask the OS for a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _normalize_worker_url(raw_url: str) -> str:
    """Strip trailing ``/v1`` and trailing slashes from an upstream URL.

    The gateway client always sends ``api_path="/v1"`` when registering a
    worker, and the upstream's ``api_url`` is computed as
    ``url + api_path``. Without this normalization, callers passing an
    OpenAI-compatible URL like ``http://localhost:4000/v1`` would end up
    forwarding to ``http://localhost:4000/v1/v1/chat/completions``.
    """
    url = raw_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _get_routable_ip() -> str:
    """Return the machine's routable IPv4 address.

    Strategy (adapted from slime's ``get_host_info``):
    1. UDP probe to 8.8.8.8 — queries kernel routing table without sending data
    2. Fallback: ``socket.getaddrinfo(hostname)`` filtering out loopback
    3. Last resort: ``127.0.0.1``
    """

    def _is_loopback(ip: str) -> bool:
        return ip.startswith("127.") or ip == "::1"

    # Strategy 1: UDP connect probe (most accurate)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip: str = s.getsockname()[0]
            if not _is_loopback(ip):
                return ip
    except Exception:
        pass

    # Strategy 2: hostname resolution filtering out loopback
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in infos:
            ip = str(info[4][0])
            if not _is_loopback(ip):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


class GatewayManager:
    """Manages model gateway lifecycle for training.

    Supports two execution modes:
    - 'process': subprocess.Popen (for verl / distributed)
    - 'thread': background thread via create_app + uvicorn (for tinker / single-machine)
    """

    # Subclasses override these to flip vLLM-specific request-body injection
    # off when the upstream isn't vLLM (e.g. OpenAI/Anthropic via LiteLLM).
    add_logprobs: bool = True
    add_return_token_ids: bool = True

    def __init__(self, config: DictConfig, mode: str = "thread") -> None:
        gw_cfg = config.rllm.get("gateway", {})
        configured_host = gw_cfg.get("host", None)
        self.host: str = configured_host if configured_host else _get_routable_ip()
        self.port: int = gw_cfg.get("port", 9090)
        self.db_path: str | None = gw_cfg.get("db_path", None)
        self.public_url: str | None = gw_cfg.get("public_url", None)
        self.tunnel_backend: str | None = gw_cfg.get("tunnel", None)
        self.sampling_params_priority: str = gw_cfg.get("sampling_params_priority", "client")
        # The gateway always pins ``body.model`` to whatever the trainer is serving
        self.model: str | None = config.get("model", {}).get("name", None)
        self.mode = mode

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server when using thread mode
        self._local_handler: Any = None  # in-process handler for tinker
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None
        self._tunnel: Any = None  # CloudflaredTunnel when tunnel_backend is set

        # Per-mode sampling params (extracted from rollout engine in start())
        self._train_sampling_params: dict[str, Any] = {}
        self._val_sampling_params: dict[str, Any] = {}

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def client(self) -> GatewayClient:
        """Sync client for lifecycle operations (start, stop, health polling)."""
        if self._client is None:
            self._client = GatewayClient(self.gateway_url)
        return self._client

    @property
    def async_client(self) -> AsyncGatewayClient:
        """Async client for runtime operations (sessions, traces)."""
        if self._async_client is None:
            self._async_client = AsyncGatewayClient(self.gateway_url, timeout=_TRACE_API_TIMEOUT)
        return self._async_client

    # -- Lifecycle -----------------------------------------------------------

    def start(self, rollout_engine: RolloutEngine) -> None:
        """Start the gateway and register inference workers.

        For VerlEngine: registers the existing vLLM server addresses.
        For TinkerEngine: creates an in-process handler (no sidecar needed).
        """
        engine_cls = type(rollout_engine).__name__

        if engine_cls == "TinkerEngine":
            # In-process handler — no HTTP backend, no worker registration
            from rllm.experimental.engine.tinker_adapter import create_tinker_handler

            self._local_handler = create_tinker_handler(rollout_engine)
            self._start_thread(local_handler=self._local_handler)
        elif engine_cls == "VerlEngine":
            if self.mode == "process":
                self._start_process()
            else:
                self._start_thread()

            worker_urls = self._ensure_verl_engine_workers(rollout_engine)
            for url in worker_urls:
                worker_id = self.client.add_worker(url=url)
                logger.info("Registered worker %s -> %s", worker_id, url)
        else:
            logger.warning("Unknown engine type %s — no workers registered", engine_cls)

        # Extract per-mode sampling params from the rollout engine
        self._train_sampling_params = getattr(rollout_engine, "train_sampling_params", {})
        self._val_sampling_params = getattr(rollout_engine, "val_sampling_params", {})

        # Bring up a tunnel after the gateway is healthy. Remote
        # sandboxes (Modal/Daytona/E2B/...) can't reach loopback, so
        # AgentTrainer auto-sets ``rllm.gateway.tunnel="cloudflared"``
        # when ``sandbox_backend`` isn't local. The harness's session URL
        # threads through ``self.public_url`` (see ``get_session_url``),
        # so flipping ``self.public_url`` here makes every subsequent
        # rollout use the tunneled URL automatically.
        if self.tunnel_backend and not self.public_url:
            self._start_tunnel()

    def _start_tunnel(self) -> None:
        """Start a public-URL tunnel pointing at the gateway."""
        backend = (self.tunnel_backend or "").lower()
        if backend != "cloudflared":
            raise ValueError(f"Unsupported gateway tunnel backend: {self.tunnel_backend!r}. Supported: 'cloudflared'.")

        from rllm.experimental.engine.tunnel import CloudflaredTunnel

        tunnel = CloudflaredTunnel(self.gateway_url)
        self.public_url = tunnel.start()
        self._tunnel = tunnel

    def stop(self) -> None:
        """Terminate the gateway (process or thread)."""
        if self._tunnel is not None:
            try:
                self._tunnel.stop()
            except Exception:
                logger.exception("Error stopping cloudflared tunnel")
            self._tunnel = None

        if self._client is not None:
            self._client.close()
            self._client = None

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._thread = None
            self._server = None

        self._local_handler = None

    # -- Session / trace API -------------------------------------------------

    def create_session(self, session_id: str, is_validation: bool = False) -> str:
        sp = self._val_sampling_params if is_validation else self._train_sampling_params
        return self.client.create_session(session_id=session_id, sampling_params=sp or None)

    def get_session_url(self, session_id: str) -> str:
        if self.public_url:
            base = self.public_url.rstrip("/")
            return f"{base}/sessions/{session_id}/v1"
        return self.client.get_session_url(session_id)

    def get_traces(self, session_id: str) -> list[TraceRecord]:
        self.client.flush()
        return self.client.get_session_traces(session_id)

    # -- Async session / trace API -------------------------------------------

    async def acreate_session(self, session_id: str, is_validation: bool = False) -> str:
        sp = self._val_sampling_params if is_validation else self._train_sampling_params
        return await self.async_client.create_session(session_id=session_id, sampling_params=sp or None)

    async def aget_traces(self, session_id: str) -> list[TraceRecord]:
        await self.async_client.flush(timeout=_TRACE_API_TIMEOUT)
        return await self.async_client.get_session_traces(session_id)

    async def adelete_session(self, session_id: str) -> int:
        """Delete a session and all its accumulated traces. Returns count removed."""
        await self.async_client.flush()
        return await self.async_client.delete_session(session_id)

    async def adelete_sessions(self, session_ids: list[str]) -> int:
        """Batch-delete many sessions in a single flush + request."""
        if not session_ids:
            return 0
        await self.async_client.flush()
        return await self.async_client.delete_sessions(session_ids)

    # -- Worker setup --------------------------------------------------------

    def _ensure_verl_engine_workers(self, rollout_engine: VerlEngine) -> list[str]:
        """Get or create worker URLs for the VerlEngine."""
        addresses = rollout_engine.server_manager._server_id_to_handle.keys()
        return [f"http://{addr}" if not addr.startswith("http") else addr for addr in addresses]

    # -- Internal ------------------------------------------------------------

    def _start_process(self) -> None:
        """Launch gateway as a subprocess and poll until healthy."""
        cmd = [
            sys.executable,
            "-m",
            "rllm_model_gateway",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
        ]
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])
        if self.sampling_params_priority != "client":
            cmd.extend(["--sampling-params-priority", self.sampling_params_priority])
        if self.model:
            cmd.extend(["--model", self.model])

        logger.info("Starting gateway subprocess: %s", " ".join(cmd))
        # Inherit parent's stdout/stderr so gateway logs are visible for debugging.
        # subprocess.PIPE causes problems as without an active reader, the OS pipe
        # buffer (~64KB on Linux) fills up under high-throughput logging, causing the
        # gateway process to block on write and eventually hang.
        self._process = subprocess.Popen(cmd)

        # Poll health endpoint
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                self.client.health()
                logger.info("Gateway process healthy at %s", self.gateway_url)
                return
            except Exception as e:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Gateway process exited unexpectedly (rc={self._process.returncode})") from e
                time.sleep(_HEALTH_POLL_INTERVAL)

        self._process.terminate()
        raise TimeoutError(f"Gateway did not become healthy within {_HEALTH_POLL_TIMEOUT}s")

    def _start_thread(self, local_handler: Any = None) -> None:
        """Start gateway in a background thread using create_app + uvicorn."""
        import uvicorn
        from rllm_model_gateway.models import GatewayConfig
        from rllm_model_gateway.server import create_app

        gw_config = GatewayConfig(
            host="0.0.0.0",
            port=self.port,
            db_path=self.db_path,
            store_worker="sqlite" if self.db_path else "memory",
            sampling_params_priority=self.sampling_params_priority,
            model=self.model,
            add_logprobs=self.add_logprobs,
            add_return_token_ids=self.add_return_token_ids,
        )
        app = create_app(config=gw_config, local_handler=local_handler)

        uvi_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(uvi_config)
        self._server = server

        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()

        # Wait for server to start
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            if server.started:
                logger.info("Gateway thread healthy at %s", self.gateway_url)
                return
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(f"Gateway thread did not start within {_HEALTH_POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# EvalGatewayManager — eval-side gateway with a static upstream URL
# ---------------------------------------------------------------------------


class EvalGatewayManager(GatewayManager):
    """Gateway pointing at a single static upstream URL (no rollout engine).

    Used by ``rllm eval``: the upstream is either the user's ``--base-url``
    (vLLM endpoint, OpenAI-compatible server) or the URL of a LiteLLM
    proxy started by ``EvalProxyManager``.

    Key differences vs the training-side :class:`GatewayManager`:

    * vLLM-specific request-body injection (``logprobs``,
      ``return_token_ids``) is OFF — external providers reject
      ``return_token_ids`` as an unknown parameter.
    * ``start()`` ignores any ``rollout_engine`` and registers the
      upstream URL passed at construction. URLs are normalized via
      :func:`_normalize_worker_url` (strips trailing ``/v1``).

    Example::

        gw = EvalGatewayManager(upstream_url=base_url, model="gpt-4o")
        gw.start()
        try:
            ...
        finally:
            gw.stop()
    """

    add_logprobs = False
    add_return_token_ids = False

    def __init__(
        self,
        upstream_url: str,
        model: str,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        db_path: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "rllm": {
                    "gateway": {
                        "host": host,
                        "port": port if port is not None else _find_free_port(),
                        "db_path": db_path,
                    }
                },
                "model": {"name": model},
            }
        )
        super().__init__(cfg, mode="thread")
        self._upstream_urls: list[str] = [upstream_url]

    def start(self, rollout_engine: RolloutEngine | None = None) -> None:  # type: ignore[override]
        """Start gateway and register the static upstream URL(s).

        ``rollout_engine`` is accepted for shape-compatibility with the
        base class signature but ignored — this gateway has no engine.
        """
        if rollout_engine is not None:
            logger.warning("EvalGatewayManager.start ignores `rollout_engine` argument")
        self._start_thread()
        for raw_url in self._upstream_urls:
            url = _normalize_worker_url(raw_url)
            worker_id = self.client.add_worker(url=url)
            logger.info("Registered worker %s -> %s (raw=%s)", worker_id, url, raw_url)
