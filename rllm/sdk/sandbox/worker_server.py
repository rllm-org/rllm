#!/usr/bin/env python3
"""Standalone worker server that runs inside sandboxes.

This script is self-contained — it inlines the metadata slug encoding
and result submission logic to avoid pulling in rllm.sdk.session
dependencies inside the sandbox.

Usage::

    python worker_server.py \
        --agent-module agent \
        --agent-func rollout \
        --port 8100 \
        --agent-dir /app/agent
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from urllib.parse import urlparse, urlunparse

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inlined metadata slug encoding (avoids importing rllm.sdk.session)
# ---------------------------------------------------------------------------

_SLUG_PREFIX = "rllm1:"


def _encode_metadata_slug(metadata: dict) -> str:
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
    return f"{_SLUG_PREFIX}{encoded.decode('ascii')}"


def _build_proxied_base_url(base_url: str, metadata: dict) -> str:
    slug = _encode_metadata_slug(metadata)
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    has_v1 = path.endswith("/v1")
    if has_v1:
        path = path[:-3]
    new_path = f"{path}/meta/{slug}"
    if has_v1:
        new_path += "/v1"
    if not new_path.startswith("/"):
        new_path = "/" + new_path
    rebuilt = parsed._replace(path=new_path)
    return urlunparse(rebuilt)


# ---------------------------------------------------------------------------
# Inlined result submission
# ---------------------------------------------------------------------------


async def _submit_result(proxy_base_url: str, execution_id: str, result_data: dict) -> None:
    base = proxy_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/rllm/results/{execution_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=result_data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("Failed to submit result for %s: HTTP %s – %s", execution_id, resp.status, body)
                else:
                    logger.debug("Result submitted for %s", execution_id)
    except Exception:
        logger.exception("Failed to submit result for %s", execution_id)


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------


def _load_agent_func(agent_module: str, agent_func: str, agent_dir: str | None = None):
    """Import the agent function from the user's module."""
    if agent_dir and agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    module = importlib.import_module(agent_module)
    func = getattr(module, agent_func)
    return func


# ---------------------------------------------------------------------------
# Worker server
# ---------------------------------------------------------------------------


class WorkerServer:
    def __init__(self, agent_func, port: int = 8100):
        self.agent_func = agent_func
        self.port = port
        self.app = web.Application()
        self.app.router.add_post("/execute", self.handle_execute)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_execute(self, request: web.Request) -> web.Response:
        """Accept a task, respond 202 immediately, then run in background."""
        body = await request.json()
        execution_id = body["execution_id"]
        proxy_url = body["proxy_url"]
        task = body["task"]
        agent_config = body.get("agent_config", {})

        # Fire and forget — respond 202 and process in background
        asyncio.create_task(self._run_and_push(execution_id, proxy_url, task, agent_config))
        return web.json_response({"status": "accepted", "execution_id": execution_id}, status=202)

    async def _run_and_push(self, execution_id: str, proxy_url: str, task: dict, agent_config: dict) -> None:
        """Generate session_uid, encode into proxy URL, call rollout, push result."""
        start = time.time()
        session_uid = str(uuid.uuid4())

        # Build a proxied base URL with session_uid encoded via metadata slug
        metadata = {"session_uids": [session_uid], "session_name": f"sandbox-{execution_id[:8]}"}
        proxied_url = _build_proxied_base_url(proxy_url, metadata)

        # Inject the proxied URL into the config for the agent to use
        config = dict(agent_config)
        config["base_url"] = proxied_url
        config["session_uid"] = session_uid

        result_data: dict = {}
        try:
            # Call the agent's rollout function
            if inspect.iscoroutinefunction(self.agent_func):
                trajectories = await self.agent_func(task, config)
            else:
                trajectories = await asyncio.to_thread(self.agent_func, task, config)

            # Serialize trajectories — they should be pydantic models with .model_dump()
            serialized = []
            if trajectories:
                for traj in trajectories:
                    if hasattr(traj, "model_dump"):
                        serialized.append(traj.model_dump())
                    elif isinstance(traj, dict):
                        serialized.append(traj)
                    else:
                        serialized.append({"name": "agent", "steps": [], "reward": None})

            elapsed = time.time() - start
            result_data = {
                "success": True,
                "trajectories": serialized,
                "session_uid": session_uid,
                "reward": None,
                "error": None,
                "elapsed": elapsed,
            }
        except Exception:
            elapsed = time.time() - start
            error_msg = traceback.format_exc()
            logger.error("Agent execution failed for %s: %s", execution_id, error_msg)
            result_data = {
                "success": False,
                "trajectories": None,
                "session_uid": session_uid,
                "reward": None,
                "error": error_msg,
                "elapsed": elapsed,
            }

        # Push result to the proxy's result store
        await _submit_result(proxy_url, execution_id, result_data)

    def run(self) -> None:
        web.run_app(self.app, host="0.0.0.0", port=self.port, print=logger.info)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sandbox worker server")
    parser.add_argument("--agent-module", required=True, help="Python module containing the agent function")
    parser.add_argument("--agent-func", default="rollout", help="Function name in the agent module")
    parser.add_argument("--port", type=int, default=8100, help="Port to listen on")
    parser.add_argument("--agent-dir", default=None, help="Directory to add to sys.path for agent import")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent_func = _load_agent_func(args.agent_module, args.agent_func, args.agent_dir)
    logger.info("Loaded agent function: %s.%s", args.agent_module, args.agent_func)

    server = WorkerServer(agent_func=agent_func, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
