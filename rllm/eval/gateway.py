"""EvalGatewayManager: rLLM model-gateway booted in-process for eval.

Replaces the legacy LiteLLM-based ``EvalProxyManager``. The gateway runs
as a uvicorn server in a daemon thread, so the eval CLI stays a single
process — no subprocess management, no temp YAML files.

Lifecycle mirrors the old class (``start`` / ``shutdown`` / ``get_url``)
so the CLI swap is mechanical.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

import uvicorn
from rllm_model_gateway import GatewayConfig, ProviderRoute, create_app

from rllm.eval.config import get_provider_info

logger = logging.getLogger(__name__)


def _reserve_local_port(host: str) -> int:
    """Bind+release an ephemeral TCP port to avoid collisions on auto-port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class EvalGatewayManager:
    """Manages an in-process rLLM gateway routing to a single provider.

    Same constructor signature as the legacy ``EvalProxyManager`` so the
    CLI swap is a one-line change.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        db_path: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.host = host
        self.port = port
        self.db_path = db_path
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._url: str = ""

    @property
    def base_url(self) -> str:
        return self._url or f"http://{self.host}:{self.port}/v1"

    def get_url(self) -> str:
        """Return the gateway's OpenAI-compatible base URL."""
        return self.base_url

    def build_config(self) -> GatewayConfig:
        """Compose a single-route ``GatewayConfig`` from the provider registry.

        The API key is exported into ``info.env_key`` so the gateway can read
        it via ``ProviderRoute.api_key_env`` — it is never stored on the
        config object.
        """
        info = get_provider_info(self.provider)
        if info is None or not info.backend_url:
            raise ValueError(f"Provider '{self.provider}' has no configured backend_url. Use the 'custom' provider with --base-url for unsupported endpoints.")
        env_key = info.env_key or f"RLLM_PROVIDER_{self.provider.upper()}_KEY"
        os.environ[env_key] = self.api_key

        return GatewayConfig(
            host=self.host,
            port=self.port,
            db_path=self.db_path,
            store_worker="sqlite" if self.db_path else "memory",
            sync_traces=False,
            add_logprobs=False,
            add_return_token_ids=False,
            strip_vllm_fields=False,
            providers=[
                ProviderRoute(
                    model_name=self.model_name,
                    backend_url=info.backend_url,
                    backend_model=self.model_name,
                    api_key_env=env_key,
                )
            ],
        )

    def start(self, *, startup_timeout: float = 10.0) -> str:
        """Boot the gateway in a background thread and return its URL."""
        if self._server is not None:
            return self.base_url

        if self.port == 0:
            self.port = _reserve_local_port(self.host)

        config = self.build_config()
        config.port = self.port

        app = create_app(config)
        uv_config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(uv_config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="rllm-gateway")
        self._thread.start()

        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._server.started:
                self._url = f"http://{self.host}:{self.port}/v1"
                logger.info("rLLM gateway ready at %s", self._url)
                return self._url
            time.sleep(0.05)

        # Boot failed — clean up before raising.
        self.shutdown()
        raise RuntimeError(f"rLLM gateway did not start within {startup_timeout}s")

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __repr__(self) -> str:
        return f"EvalGatewayManager(provider={self.provider}, model={self.model_name}, url={self.base_url})"
