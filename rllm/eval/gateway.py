"""EvalGatewayManager: rLLM model-gateway booted in-process for eval.

Replaces the legacy LiteLLM-based ``EvalProxyManager``. The gateway runs
as a uvicorn server in a daemon thread, so the eval CLI stays a single
process — no subprocess management, no temp YAML files.

All gateway runs share a single sqlite trace store at
``~/.rllm/gateway/traces.db`` (override via ``RLLM_GATEWAY_DB``). Each
run is tagged with its own ``run_id`` (typically the eval run dir
basename) so concurrent invocations don't collide on session ids and
the cross-run viewer can group activity by run.

Lifecycle mirrors the old class (``start`` / ``shutdown`` / ``get_url``)
so the CLI swap is mechanical.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Any

import uvicorn
from rllm_model_gateway import GatewayConfig, ProviderRoute, create_app

from rllm.eval.config import get_provider_info

logger = logging.getLogger(__name__)

# Single shared sqlite for every gateway invocation. Concurrent writers
# are serialised by sqlite's WAL + busy_timeout, no extra locking needed
# (LLM call rates are well below sqlite's contention threshold).
_DEFAULT_GATEWAY_DB = "~/.rllm/gateway/traces.db"


def _default_gateway_db_path() -> str:
    """Resolve the shared db path: env var → default."""
    return os.path.expanduser(os.environ.get("RLLM_GATEWAY_DB", _DEFAULT_GATEWAY_DB))


def _reserve_local_port(host: str) -> int:
    """Bind+release an ephemeral TCP port to avoid collisions on auto-port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class EvalGatewayManager:
    """Manages an in-process rLLM gateway routing to a single provider.

    Same constructor signature as the legacy ``EvalProxyManager`` so the
    CLI swap is a one-line change. Adds ``run_id`` + ``run_metadata`` so
    the gateway can tag every persisted trace with the run identity.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        db_path: str | None = None,
        run_id: str | None = None,
        run_metadata: dict[str, Any] | None = None,
        public_url: str | None = None,
        auto_tunnel: bool = False,
    ) -> None:
        import secrets

        if public_url and auto_tunnel:
            raise ValueError("public_url and auto_tunnel are mutually exclusive")

        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        # ``publicly_exposed`` covers both "user supplied a public URL"
        # and "we're going to spawn a tunnel after gateway start" — the
        # bind interface and inbound-auth requirement are the same.
        publicly_exposed = bool(public_url) or auto_tunnel
        if publicly_exposed and host == "127.0.0.1":
            host = "0.0.0.0"
        self.host = host
        self.port = port
        # Default to the shared per-user gateway db so traces from
        # multiple eval runs land in one place — matches the cross-run
        # viewer's data model.
        self.db_path = db_path or _default_gateway_db_path()
        self.run_id = run_id
        self.run_metadata: dict[str, Any] = dict(run_metadata or {})
        self.public_url = public_url  # may be set later by ``start()`` when auto_tunnel
        self._auto_tunnel = auto_tunnel
        # Inbound auth: required whenever the gateway is exposed beyond
        # the local host. Loopback-only deployments (in-process eval
        # against docker/local) skip the auth check for backwards
        # compatibility — there's no attack surface there.
        self.inbound_auth_token: str | None = secrets.token_urlsafe(32) if publicly_exposed else None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._url: str = ""
        self._tunnel_proc = None  # subprocess.Popen, set when auto_tunnel=True

    @property
    def base_url(self) -> str:
        # When the user supplied a public URL (tunnel, public IP, …) hand
        # *that* to harnesses so cloud sandboxes can reach the gateway.
        # The locally-bound URL is still used for in-process readiness.
        if self.public_url:
            return self._normalize_v1(self.public_url)
        return self._url or f"http://{self.host}:{self.port}/v1"

    @property
    def local_url(self) -> str:
        """The URL the gateway is bound to on this host (no public override)."""
        return self._url or f"http://{self.host}:{self.port}/v1"

    @staticmethod
    def _normalize_v1(url: str) -> str:
        """Ensure the URL ends in ``/v1`` so eval runner's session stamping aligns."""
        u = url.rstrip("/")
        return u if u.endswith("/v1") else f"{u}/v1"

    def get_url(self) -> str:
        """Return the gateway's OpenAI-compatible base URL (public if set)."""
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

        # OpenAI's o-series + GPT-5 models reject ``max_tokens`` and require
        # ``max_completion_tokens``. CLI agents (opencode, codex, mini-swe-agent)
        # all send ``max_tokens`` by default. Dropping it lets the model use
        # its own default output budget — the cleanest workaround until we
        # extend the gateway with a rename map. Older models (gpt-4-turbo,
        # gpt-3.5) still accept this; they just fall back to defaults too.
        drop_params = ["max_tokens"] if self.provider == "openai" else []

        return GatewayConfig(
            host=self.host,
            port=self.port,
            db_path=self.db_path,
            store_worker="sqlite",
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
                    drop_params=drop_params,
                )
            ],
            run_id=self.run_id,
            run_metadata=self.run_metadata,
            inbound_auth_token=self.inbound_auth_token,
        )

    @property
    def _cleanup_name(self) -> str:
        """Stable per-instance key for the late-cleanup registry."""
        return f"gateway-{id(self)}"

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
                # Local URL is what readiness probes reach; if a public URL
                # was supplied (or auto-tunneled below), ``base_url`` will
                # return that instead so harnesses pass it to in-sandbox CLIs.
                self._url = f"http://{self.host}:{self.port}/v1"
                if self._auto_tunnel and self.public_url is None:
                    self._spawn_tunnel()
                if self.public_url:
                    logger.info("rLLM gateway bound at %s, exposed via %s", self._url, self.public_url)
                else:
                    logger.info("rLLM gateway ready at %s", self._url)
                # Register so SIGTERM / atexit teardown reaches us even
                # when the eval CLI's outer ``finally`` doesn't run
                # (e.g., process killed mid-rollout).
                from rllm.sandbox.cleanup import register_late_cleanup

                register_late_cleanup(self._cleanup_name, self.shutdown)
                return self.base_url
            time.sleep(0.05)

        # Boot failed — clean up before raising.
        self.shutdown()
        raise RuntimeError(f"rLLM gateway did not start within {startup_timeout}s")

    def _spawn_tunnel(self) -> None:
        """Spawn cloudflared and pin the resulting public URL onto self.

        Called from :meth:`start` after the gateway is bound, so the
        tunnel has a port to forward to. Failures here propagate so
        callers see the misconfiguration immediately rather than
        learning about it via mysterious 404s from the sandbox.
        """
        from rllm.eval.tunnel import start_cloudflared_tunnel

        url, proc = start_cloudflared_tunnel(self.port)
        self.public_url = url
        self._tunnel_proc = proc

    def shutdown(self) -> None:
        # Deregister from the cleanup registry first so that if the
        # registry's close_all() is mid-iteration on us, our own
        # ``shutdown`` calls don't recurse. Idempotent.
        from rllm.sandbox.cleanup import deregister_late_cleanup

        deregister_late_cleanup(self._cleanup_name)
        # Tunnel first: keeping it alive after the gateway is gone
        # leaves a public URL pointing at a dead port.
        if self._tunnel_proc is not None:
            from rllm.eval.tunnel import stop_tunnel

            stop_tunnel(self._tunnel_proc)
            self._tunnel_proc = None
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __repr__(self) -> str:
        return f"EvalGatewayManager(provider={self.provider}, model={self.model_name}, url={self.base_url})"
