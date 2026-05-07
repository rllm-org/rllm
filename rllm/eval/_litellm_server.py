#!/usr/bin/env python3
"""Minimal LiteLLM proxy launcher used by ``EvalProxyManager``.

Starts a LiteLLM proxy that forwards OpenAI Chat Completions requests to a
configured external provider (OpenAI, Anthropic, Together, ...). Exposes a
single ``/admin/reload`` endpoint so :class:`rllm.eval.proxy.EvalProxyManager`
can push a config after the process is up.

This is *not* a tracing or sampling-injection proxy — for training we use the
``rllm-model-gateway`` instead. Eval against external providers is the only
reason this LiteLLM proxy exists.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import litellm
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from litellm.proxy.proxy_server import app as litellm_app
from litellm.proxy.proxy_server import initialize
from pydantic import BaseModel, Field


class ReloadPayload(BaseModel):
    config_yaml: str = Field(description="Inline LiteLLM config YAML.")


class _Runtime:
    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._current_config: Path | None = None
        self._lock = asyncio.Lock()

    async def reload(self, payload: ReloadPayload) -> Path:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        target = self._state_dir / "litellm_proxy_config_autogen.yaml"
        target.write_text(payload.config_yaml)

        async with self._lock:
            if self._current_config is not None:
                if hasattr(litellm, "model_list"):
                    litellm.model_list = []
                if hasattr(litellm, "router"):
                    litellm.router = None
                litellm.callbacks = []

            os.environ["LITELLM_CONFIG"] = str(target)
            litellm.drop_params = True
            await initialize(config=str(target), telemetry=False)
            self._current_config = target
        return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteLLM proxy server for rllm eval.")
    parser.add_argument("--host", default=os.getenv("LITELLM_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LITELLM_PROXY_PORT", "4000")))
    parser.add_argument("--state-dir", default=os.getenv("LITELLM_PROXY_STATE_DIR", "./.litellm_proxy"))
    parser.add_argument("--admin-token", default=os.getenv("LITELLM_PROXY_ADMIN_TOKEN"))
    parser.add_argument("--log-level", default=os.getenv("LITELLM_PROXY_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runtime = _Runtime(state_dir=Path(args.state_dir).expanduser().resolve())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    def _require_token(authorization: str = Header(default="")) -> None:
        if args.admin_token and authorization != f"Bearer {args.admin_token}":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")

    @litellm_app.post("/admin/reload", dependencies=[Depends(_require_token)])
    async def reload_proxy(payload: ReloadPayload):
        try:
            new_path = await runtime.reload(payload)
            return {"status": "reloaded", "config_path": str(new_path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("Reload failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    def _shutdown_handler(*_: int) -> None:
        raise SystemExit

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    uvicorn.run(litellm_app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
