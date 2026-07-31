from __future__ import annotations

import asyncio
import logging
import multiprocessing
import secrets
import time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from rllm_model_gateway.v2 import (
    AsyncGatewayClient,
    GatewayClient,
    GatewayConfig,
    InferenceClientClass,
    create_app,
)

from rllm.gateway.manager import _get_routable_ip
from rllm.gateway.tunnel import parse_tunnel
from rllm.gateway.types import GatewaySession

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_TRACE_API_TIMEOUT = 600.0


def _run_gateway(
    config_data: dict[str, Any],
    inference_client_cls: InferenceClientClass,
    inference_client_kwargs: dict[str, Any],
    gateway_connection: Any,
) -> None:
    import uvicorn

    config = GatewayConfig.model_validate(config_data)
    server: uvicorn.Server | None = None

    def shutdown() -> None:
        assert server is not None
        server.should_exit = True

    app = create_app(
        config,
        inference_client_cls,
        inference_client_kwargs,
        gateway_connection,
        shutdown,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=config.host, port=config.port, log_level="warning"))
    server.run()


class GatewayManagerV2:
    def __init__(self, config: DictConfig) -> None:
        configured_host = config.get("host", None)
        self.host = configured_host if configured_host else _get_routable_ip()
        tokenizer_model = config.get("tokenizer_model")
        if not tokenizer_model:
            raise ValueError("rllm.gateway.tokenizer_model is required for gateway.version=v2")
        configured_workers = config.get("num_workers", None)
        self._gateway_config = GatewayConfig(
            host="0.0.0.0",
            port=int(config.get("port", 9090)),
            num_workers=1 if configured_workers is None else int(configured_workers),
            worker_startup_timeout_seconds=float(config.get("worker_startup_timeout_seconds", 300.0)),
            request_timeout_seconds=float(config.get("request_timeout_seconds", 3600.0)),
            update_timeout_seconds=float(config.get("update_timeout_seconds", 300.0)),
            heartbeat_initial_delay_seconds=float(config.get("heartbeat_initial_delay_seconds", 60.0)),
            heartbeat_interval_seconds=float(config.get("heartbeat_interval_seconds", 10.0)),
            admin_key=secrets.token_urlsafe(32),
            cumulative=bool(config.get("cumulative_token_mode", False)),
            tokenizer_model=str(tokenizer_model),
            renderer=str(config.get("renderer_family", "auto")),
            renderer_kwargs=dict(config.get("renderer_kwargs", {})),
        )
        self.public_url, self.tunnel_backend = parse_tunnel(config.get("tunnel", None))
        self._process: BaseProcess | None = None
        self._client: GatewayClient | None = None
        self._async_client: AsyncGatewayClient | None = None
        self._tunnel: Any = None
        self._trainer_connection: Any = None

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self._gateway_config.port}"

    @property
    def client(self) -> GatewayClient:
        if self._client is None:
            self._client = GatewayClient(self.gateway_url, self._gateway_config.admin_key, timeout=_TRACE_API_TIMEOUT)
        return self._client

    @property
    def async_client(self) -> AsyncGatewayClient:
        if self._async_client is None:
            self._async_client = AsyncGatewayClient(self.gateway_url, self._gateway_config.admin_key, timeout=_TRACE_API_TIMEOUT)
        return self._async_client

    def start(
        self,
        inference_client_cls: InferenceClientClass,
        inference_client_kwargs: dict[str, Any],
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._trainer_connection, gateway_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_run_gateway,
            args=(
                self._gateway_config.model_dump(mode="json"),
                inference_client_cls,
                inference_client_kwargs,
                gateway_connection,
            ),
            name="rllm-gateway-head",
        )
        try:
            self._process.start()
        except Exception:
            gateway_connection.close()
            self._trainer_connection.close()
            self._trainer_connection = None
            self._process = None
            raise
        gateway_connection.close()
        try:
            self._wait_until_healthy()
            if self.tunnel_backend:
                self._start_tunnel()
        except Exception:
            self.stop()
            raise

    async def update_inference_client(self, update: dict[str, Any]) -> None:
        if self._process is None or self._trainer_connection is None:
            raise RuntimeError("gateway is not running")
        await asyncio.to_thread(self._trainer_connection.send, update)
        deadline = time.monotonic() + self._gateway_config.update_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("gateway inference client update timed out")
            if not await asyncio.to_thread(self._trainer_connection.poll, min(0.5, remaining)):
                if not self._process.is_alive():
                    raise RuntimeError(f"gateway process exited during inference client update (code={self._process.exitcode})")
                continue
            try:
                response = await asyncio.to_thread(self._trainer_connection.recv)
            except EOFError as exc:
                raise RuntimeError("gateway control pipe closed during inference client update") from exc
            if not response.get("ok"):
                raise RuntimeError(f"gateway inference client update failed: {response.get('error', 'unknown error')}")
            return

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
        shutdown_requested = False
        if self._trainer_connection is not None:
            try:
                self._trainer_connection.send(None)
                shutdown_requested = True
            except Exception:
                pass
        if self._process is not None:
            if shutdown_requested:
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process = None
        if self._trainer_connection is not None:
            self._trainer_connection.close()
            self._trainer_connection = None

    def create_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession:
        credentials = self.client.create_session(session_id=session_id, sampling_params=sampling_params)
        return GatewaySession(session_id=credentials.session_id, api_key=credentials.agent_key)

    async def acreate_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession:
        credentials = await self.async_client.create_session(session_id=session_id, sampling_params=sampling_params)
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

    def _start_tunnel(self) -> None:
        from rllm.gateway.tunnel import create_tunnel

        tunnel = create_tunnel(self.tunnel_backend, self.gateway_url)
        self.public_url = tunnel.start()
        self._tunnel = tunnel

    def _wait_until_healthy(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self._gateway_config.worker_startup_timeout_seconds
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
        raise TimeoutError(f"Gateway did not become healthy within {self._gateway_config.worker_startup_timeout_seconds}s") from last_error
