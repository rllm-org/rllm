from __future__ import annotations

import logging
import multiprocessing
import secrets
import time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from rllm_model_gateway.v2 import (
    AsyncGatewayClient,
    BackendConfig,
    GatewayClient,
    GatewayConfig,
    TokenizationConfig,
    create_app,
)

from rllm.gateway.manager import _get_routable_ip
from rllm.gateway.tunnel import parse_tunnel
from rllm.gateway.types import GatewaySession

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.engine.rollout import RolloutEngine

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_TRACE_API_TIMEOUT = 600.0


def _run_gateway(config_data: dict[str, Any]) -> None:
    import uvicorn

    config = GatewayConfig.model_validate(config_data)
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="warning")


class GatewayManagerV2:
    def __init__(self, config: DictConfig) -> None:
        gateway_config = config.rllm.get("gateway", {})
        configured_host = gateway_config.get("host", None)
        self.host = configured_host if configured_host else _get_routable_ip()
        self.port = int(gateway_config.get("port", 9090))
        configured_workers = gateway_config.get("num_workers", None)
        self.num_workers = 1 if configured_workers is None else int(configured_workers)
        if self.num_workers < 1:
            raise ValueError("rllm.gateway.num_workers must be at least 1 for gateway.version=v2")

        self.worker_startup_timeout_seconds = float(gateway_config.get("worker_startup_timeout_seconds", 300.0))
        self.request_timeout_seconds = float(gateway_config.get("request_timeout_seconds", 3600.0))
        self.heartbeat_seconds = float(gateway_config.get("heartbeat_seconds", 10.0))
        self.cumulative = bool(gateway_config.get("cumulative_token_mode", False))

        model_config = config.get("model", {})
        tokenizer_model = model_config.get("tokenizer_model") or model_config.get("name")
        if not tokenizer_model:
            raise ValueError("model.tokenizer_model or model.name is required for gateway.version=v2")
        self.tokenization = TokenizationConfig(
            model=str(tokenizer_model),
            renderer=str(gateway_config.get("renderer_family", "auto")),
            renderer_kwargs=dict(gateway_config.get("renderer_kwargs", {})),
        )

        self.public_url, self.tunnel_backend = parse_tunnel(gateway_config.get("tunnel", None))
        self._admin_key = secrets.token_urlsafe(32)
        self._process: BaseProcess | None = None
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None
        self._tunnel: Any = None
        self._train_sampling_params: dict[str, Any] = {}
        self._val_sampling_params: dict[str, Any] = {}

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def client(self) -> GatewayClient:
        if self._client is None:
            self._client = GatewayClient(self.gateway_url, self._admin_key, timeout=_TRACE_API_TIMEOUT)
        return self._client

    @property
    def async_client(self) -> AsyncGatewayClient:
        if self._async_client is None:
            self._async_client = AsyncGatewayClient(self.gateway_url, self._admin_key, timeout=_TRACE_API_TIMEOUT)
        return self._async_client

    def start(self, rollout_engine: RolloutEngine) -> None:
        backend_data = rollout_engine.gateway_backend_config()
        if not isinstance(backend_data, dict):
            raise TypeError("RolloutEngine.gateway_backend_config() must return a dict")
        backend = BackendConfig.model_validate(backend_data)

        self._train_sampling_params = getattr(rollout_engine, "train_sampling_params", {})
        self._val_sampling_params = getattr(rollout_engine, "val_sampling_params", {})
        config = GatewayConfig(
            host="0.0.0.0",
            port=self.port,
            num_workers=self.num_workers,
            worker_startup_timeout_seconds=self.worker_startup_timeout_seconds,
            request_timeout_seconds=self.request_timeout_seconds,
            heartbeat_seconds=self.heartbeat_seconds,
            admin_key=self._admin_key,
            cumulative=self.cumulative,
            tokenization=self.tokenization,
            backend=backend,
        )
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(
            target=_run_gateway,
            args=(config.model_dump(mode="json"),),
            name="rllm-gateway-head",
        )
        self._process.start()
        try:
            self._wait_until_healthy()
            if self.tunnel_backend:
                self._start_tunnel()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._tunnel is not None:
            try:
                self._tunnel.stop()
            except Exception:
                logger.exception("Error stopping gateway tunnel")
            self._tunnel = None
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process = None

    def create_session(
        self,
        session_id: str,
        is_validation: bool = False,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession:
        params = self._sampling_params(is_validation, sampling_params)
        credentials = self.client.create_session(session_id=session_id, sampling_params=params)
        return GatewaySession(session_id=credentials.session_id, api_key=credentials.agent_key)

    async def acreate_session(
        self,
        session_id: str,
        is_validation: bool = False,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession:
        params = self._sampling_params(is_validation, sampling_params)
        credentials = await self.async_client.create_session(session_id=session_id, sampling_params=params)
        return GatewaySession(session_id=credentials.session_id, api_key=credentials.agent_key)

    def get_traces(self, session_id: str) -> list[Any]:
        return self.client.get_session(session_id).traces

    async def aget_traces(self, session_id: str) -> list[Any]:
        return (await self.async_client.get_session(session_id)).traces

    async def adelete_session(self, session_id: str) -> None:
        await self.async_client.delete_session(session_id)

    def get_session_url(self, session_id: str, *, public: bool = True) -> str:
        base_url = self.public_url if public and self.public_url else self.gateway_url
        return f"{base_url.rstrip('/')}/sessions/{quote(session_id, safe='/')}/v1"

    async def astop(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
        self.stop()

    def _sampling_params(
        self,
        is_validation: bool,
        sampling_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if sampling_params is not None:
            return sampling_params or None
        params = self._val_sampling_params if is_validation else self._train_sampling_params
        return params or None

    def _start_tunnel(self) -> None:
        from rllm.gateway.tunnel import create_tunnel

        tunnel = create_tunnel(self.tunnel_backend, self.gateway_url)
        self.public_url = tunnel.start()
        self._tunnel = tunnel

    def _wait_until_healthy(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.worker_startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.client.health()
                logger.info("Gateway process healthy at %s", self.gateway_url)
                return
            except Exception as exc:
                last_error = exc
                if not self._process.is_alive():
                    raise RuntimeError(f"Gateway process exited unexpectedly (code={self._process.exitcode})") from exc
                time.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(f"Gateway did not become healthy within {self.worker_startup_timeout_seconds}s") from last_error
