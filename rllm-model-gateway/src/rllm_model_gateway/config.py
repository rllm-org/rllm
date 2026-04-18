from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GatewayConfig(BaseModel):
    # 127.0.0.1 by default — gateway is loopback-only unless explicitly
    # opened. Production deployments should pass host="0.0.0.0" plus run
    # behind TLS termination.
    host: str = "127.0.0.1"
    port: int = 9090
    db_path: str | None = None
    log_level: str = "INFO"

    # Upstream URL for passthrough mode. Should follow the SDK convention of
    # whatever you're forwarding to (e.g. "https://api.openai.com/v1" for
    # OpenAI, "https://api.anthropic.com" for Anthropic). Ignored when an
    # adapter is passed to ``create_app``.
    upstream_url: str | None = None

    # API key sent on outbound passthrough requests. Agents authenticate to
    # the gateway with the agent_api_key; the gateway authenticates to the
    # upstream with this key.
    upstream_api_key: str | None = None

    # If set, overrides the request body's ``model`` field on every call.
    model: str | None = None

    # Conflict resolution for per-session sampling params merged into each
    # request: "client" — request wins; "session" — session wins.
    sampling_params_priority: Literal["client", "session"] = "client"

    # Connection retries on transient httpx ConnectError (passthrough mode).
    max_retries: int = 2

    # Two-tier auth. Both auto-generated at startup if left None.
    #   admin_api_key — used by the engine for /sessions, /traces, /admin/*
    #   agent_api_key — used by agents for /sessions/{sid}/v1/...
    # Either key authenticates proxy endpoints; only admin authenticates
    # management endpoints.
    admin_api_key: str | None = None
    agent_api_key: str | None = None
