"""GatewayManager: manages the rllm-model-gateway lifecycle for training.

Supports two execution modes:
- 'process': subprocess via ``rllm-model-gateway`` CLI (for verl / distributed)
- 'thread': background thread via ``create_app`` + uvicorn (for tinker / single-machine)

For Tinker backends, an in-process handler is injected into the gateway
(via ``local_handler``), avoiding the need for a separate HTTP backend server.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from rllm_model_gateway.client import GatewayClient
from rllm_model_gateway.models import TraceRecord

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.experimental.rollout import RolloutEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0


class GatewayManager:
    """Manages model gateway lifecycle for training.

    Supports two execution modes:
    - 'process': subprocess.Popen (for verl / distributed)
    - 'thread': background thread via create_app + uvicorn (for tinker / single-machine)
    """

    def __init__(self, config: DictConfig, mode: str = "thread") -> None:
        gw_cfg = config.rllm.get("gateway", {})
        self.host: str = gw_cfg.get("host", "127.0.0.1")
        self.port: int = gw_cfg.get("port", 9090)
        self.db_path: str | None = gw_cfg.get("db_path", None)
        self.mode = mode

        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server when using thread mode
        self._local_handler: Any = None  # in-process handler for tinker
        self._client: GatewayClient | None = None

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def client(self) -> GatewayClient:
        if self._client is None:
            self._client = GatewayClient(self.gateway_url)
        return self._client

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

    def create_session(self, session_id: str) -> str:
        return self.client.create_session(session_id=session_id)

    def get_session_url(self, session_id: str) -> str:
        return self.client.get_session_url(session_id)

    def get_traces(self, session_id: str) -> list[TraceRecord]:
        self.client.flush()
        return self.client.get_session_traces(session_id)

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
            self.host,
            "--port",
            str(self.port),
        ]
        if self.db_path:
            cmd.extend(["--db-path", self.db_path])

        logger.info("Starting gateway subprocess: %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Poll health endpoint
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            try:
                self.client.health()
                logger.info("Gateway process healthy at %s", self.gateway_url)
                return
            except Exception:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    raise RuntimeError(f"Gateway process exited unexpectedly: {stderr}") from None
                time.sleep(_HEALTH_POLL_INTERVAL)

        self._process.terminate()
        raise TimeoutError(f"Gateway did not become healthy within {_HEALTH_POLL_TIMEOUT}s")

    def _start_thread(self, local_handler: Any = None) -> None:
        """Start gateway in a background thread using create_app + uvicorn."""
        import uvicorn
        from rllm_model_gateway.models import GatewayConfig
        from rllm_model_gateway.server import create_app

        gw_config = GatewayConfig(
            host=self.host,
            port=self.port,
            db_path=self.db_path,
            store_worker="sqlite" if self.db_path else "memory",
        )
        app = create_app(config=gw_config, local_handler=local_handler)

        uvi_config = uvicorn.Config(
            app,
            host=self.host,
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
