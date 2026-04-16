"""GatewayManager: manages the rllm-model-gateway lifecycle for training.

Supports two execution modes:
- 'process': subprocess via ``rllm-model-gateway`` CLI (for verl / distributed)
- 'thread': background thread via ``create_app`` + uvicorn (for tinker / single-machine)

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

    from rllm.experimental.rollout import RolloutEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0
_TRACE_API_TIMEOUT = 600.0


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

    def __init__(self, config: DictConfig, mode: str = "thread") -> None:
        gw_cfg = config.rllm.get("gateway", {})
        configured_host = gw_cfg.get("host", None)
        self.host: str = configured_host if configured_host else _get_routable_ip()
        self.port: int = gw_cfg.get("port", 9090)
        self.db_path: str | None = gw_cfg.get("db_path", None)
        self.mode = mode

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server when using thread mode
        self._local_handler: Any = None  # in-process handler for tinker
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None

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
        else:
            if self.mode == "process":
                self._start_process()
            else:
                self._start_thread()

            worker_urls = self._ensure_workers(rollout_engine)
            for url in worker_urls:
                worker_id = self.client.add_worker(url=url)
                logger.info("Registered worker %s -> %s", worker_id, url)

        # Extract per-mode sampling params from the rollout engine
        self._train_sampling_params = getattr(rollout_engine, "train_sampling_params", {})
        self._val_sampling_params = getattr(rollout_engine, "val_sampling_params", {})

    def stop(self) -> None:
        """Terminate the gateway (process or thread)."""
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
        """Delete a session and its traces from the gateway DB."""
        return await self.async_client.delete_session(session_id)

    def delete_session(self, session_id: str) -> int:
        """Delete a session and its traces from the gateway DB (sync)."""
        return self.client.delete_session(session_id)

    # -- Worker setup --------------------------------------------------------

    def _ensure_workers(self, rollout_engine: RolloutEngine) -> list[str]:
        """Get or create worker URLs for the given engine type."""
        engine_cls = type(rollout_engine).__name__

        if engine_cls == "VerlEngine":
            addresses = rollout_engine.rollout_manager.server_addresses
            return [f"http://{addr}" if not addr.startswith("http") else addr for addr in addresses]

        logger.warning("Unknown engine type %s — no workers registered", engine_cls)
        return []

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
