"""GatewayManager: starts the rllm-model-gateway in a background thread,
exposes a sync + async client to the engine, and surfaces the agent API key
so engines can hand it down to agent flows.

The gateway is always run in adapter mode: a generic engine adapter built
from the rollout engine handles every request. There is no separate HTTP
backend, no worker registration.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

from rllm_model_gateway import (
    AsyncGatewayClient,
    GatewayClient,
    GatewayConfig,
    TraceRecord,
    create_app,
)

from rllm.experimental.engine.engine_adapter import create_engine_adapter

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.experimental.rollout import RolloutEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0


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
    """Manages model gateway lifecycle for training."""

    def __init__(self, config: DictConfig) -> None:
        gw_cfg = config.rllm.get("gateway", {})
        configured_host = gw_cfg.get("host", None)
        self.host: str = configured_host if configured_host else _get_routable_ip()
        self.port: int = gw_cfg.get("port", 9090)
        self.db_path: str | None = gw_cfg.get("db_path", None)
        self.public_url: str | None = gw_cfg.get("public_url", None)
        self.sampling_params_priority: str = gw_cfg.get("sampling_params_priority", "client")
        # Engine pins body.model so agents can pass anything they like.
        self.model: str | None = config.get("model", {}).get("name", None)

        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server
        self._gateway_config: GatewayConfig | None = None
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None

        # Per-mode sampling params (extracted from rollout engine in start())
        self._train_sampling_params: dict[str, Any] = {}
        self._val_sampling_params: dict[str, Any] = {}

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def admin_api_key(self) -> str:
        if self._gateway_config is None or self._gateway_config.admin_api_key is None:
            raise RuntimeError("GatewayManager not started")
        return self._gateway_config.admin_api_key

    @property
    def agent_api_key(self) -> str:
        if self._gateway_config is None or self._gateway_config.agent_api_key is None:
            raise RuntimeError("GatewayManager not started")
        return self._gateway_config.agent_api_key

    @property
    def client(self) -> GatewayClient:
        if self._client is None:
            self._client = GatewayClient(self.gateway_url, api_key=self.admin_api_key)
        return self._client

    @property
    def async_client(self) -> AsyncGatewayClient:
        if self._async_client is None:
            self._async_client = AsyncGatewayClient(self.gateway_url, api_key=self.admin_api_key)
        return self._async_client

    # -- Lifecycle -----------------------------------------------------------

    def start(self, rollout_engine: RolloutEngine) -> None:
        """Build the engine adapter and start the gateway in a background thread."""
        adapter = create_engine_adapter(rollout_engine)

        self._gateway_config = GatewayConfig(
            host=self.host,
            port=self.port,
            db_path=self.db_path,
            model=self.model,
            sampling_params_priority=self.sampling_params_priority,
        )
        # admin_api_key / agent_api_key auto-populate inside create_app.

        import uvicorn

        app = create_app(self._gateway_config, adapter=adapter)

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

        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT
        while time.monotonic() < deadline:
            if server.started:
                logger.info(
                    "Gateway thread healthy at %s (admin_api_key=%s, agent_api_key=%s)",
                    self.gateway_url,
                    self.admin_api_key,
                    self.agent_api_key,
                )
                break
            time.sleep(_HEALTH_POLL_INTERVAL)
        else:
            raise TimeoutError(f"Gateway thread did not start within {_HEALTH_POLL_TIMEOUT}s")

        self._train_sampling_params = getattr(rollout_engine, "train_sampling_params", {}) or {}
        self._val_sampling_params = getattr(rollout_engine, "val_sampling_params", {}) or {}

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._thread = None
            self._server = None

    # -- Session / trace API -------------------------------------------------

    def create_session(self, session_id: str, is_validation: bool = False) -> str:
        sp = self._val_sampling_params if is_validation else self._train_sampling_params
        return self.client.create_session(session_id=session_id, sampling_params=sp or None)

    def get_session_url(self, session_id: str) -> str:
        if self.public_url:
            base = self.public_url.rstrip("/")
            return f"{base}/sessions/{session_id}/v1"
        return self.client.get_session_url(session_id)

    def get_traces(self, session_id: str, extras: bool = False) -> list[TraceRecord]:
        self.client.flush()
        return self.client.get_session_traces(session_id, extras=extras)

    # -- Async session / trace API -------------------------------------------

    async def acreate_session(self, session_id: str, is_validation: bool = False) -> str:
        sp = self._val_sampling_params if is_validation else self._train_sampling_params
        return await self.async_client.create_session(session_id=session_id, sampling_params=sp or None)

    async def aget_traces(self, session_id: str, extras: bool = False) -> list[TraceRecord]:
        await self.async_client.flush()
        return await self.async_client.get_session_traces(session_id, extras=extras)
