#!/usr/bin/env python3
"""Agent-side worker for AWS Bedrock AgentCore Runtimes.

This script runs inside an ACR container.  It receives invocations from the
``AgentCoreOrchestrator`` (trainer-side), calls the user's ``rollout()``
function, and POSTs results directly to the proxy's result store — identical
to ``worker_server.py`` but adapted for the ACR invocation protocol.

The script is self-contained: it inlines metadata slug encoding and result
POST logic to avoid pulling in ``rllm.sdk.session`` or ``aiohttp`` as
dependencies inside the container.

Usage (as ACR entrypoint)::

    python -m rllm.sdk.sandbox.agentcore_worker \
        --agent-module agent \
        --agent-func rollout \
        --agent-dir /app/agent

Requires the ``bedrock-agentcore`` package (``pip install bedrock-agentcore``).
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
import urllib.request
import uuid
from urllib.parse import urlparse, urlunparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined metadata slug encoding (mirrors worker_server.py)
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
# Inlined result submission (uses urllib — no aiohttp dependency needed)
# ---------------------------------------------------------------------------


_SUBMIT_MAX_RETRIES = 3
_SUBMIT_BACKOFF_BASE = 1.0  # seconds


def _submit_result(proxy_base_url: str, execution_id: str, result_data: dict) -> None:
    """POST the execution result to the proxy's result store (with retry)."""
    base = proxy_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/rllm/results/{execution_id}"

    payload = json.dumps(result_data).encode("utf-8")

    for attempt in range(_SUBMIT_MAX_RETRIES):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status < 400:
                    logger.debug("Result submitted for %s", execution_id)
                    return
                body = resp.read().decode("utf-8", errors="replace")
                logger.error(
                    "Failed to submit result for %s (attempt %d/%d): HTTP %s – %s",
                    execution_id, attempt + 1, _SUBMIT_MAX_RETRIES, resp.status, body,
                )
        except Exception:
            logger.exception(
                "Failed to submit result for %s (attempt %d/%d)",
                execution_id, attempt + 1, _SUBMIT_MAX_RETRIES,
            )

        if attempt < _SUBMIT_MAX_RETRIES - 1:
            import time
            time.sleep(_SUBMIT_BACKOFF_BASE * (2 ** attempt))

    logger.error(
        "Giving up submitting result for %s after %d attempts", execution_id, _SUBMIT_MAX_RETRIES
    )


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
# ACR entrypoint
# ---------------------------------------------------------------------------

_agent_func = None  # Set by main() before the app starts


async def _run_and_push(
    execution_id: str,
    proxy_url: str,
    task: dict,
    agent_config: dict,
    task_id: str | None = None,
) -> None:
    """Run the agent rollout and push the result to the proxy."""
    app_ref = _app  # captured for complete_async_task

    start = time.time()
    session_uid = str(uuid.uuid4())

    # Build a proxied base URL with session metadata encoded in the path
    metadata = {"session_uids": [session_uid], "session_name": f"sandbox-{execution_id[:8]}"}
    proxied_url = _build_proxied_base_url(proxy_url, metadata)

    # Inject the proxied URL into the config for the agent to use
    config = dict(agent_config)
    config["base_url"] = proxied_url
    config["session_uid"] = session_uid

    result_data: dict = {}
    try:
        assert _agent_func is not None, "Agent function not loaded"

        # Call the agent's rollout function
        if inspect.iscoroutinefunction(_agent_func):
            trajectories = await _agent_func(task, config)
        else:
            trajectories = await asyncio.to_thread(_agent_func, task, config)

        # Serialize trajectories
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
    finally:
        # Push result to the proxy's result store (synchronous, runs in thread)
        await asyncio.to_thread(_submit_result, proxy_url, execution_id, result_data)
        if task_id is not None and app_ref is not None:
            try:
                app_ref.complete_async_task(task_id)
            except Exception:
                logger.debug("complete_async_task failed (may already be done)")


_app = None  # Module-level reference, set in main()


def _create_app():
    """Create the BedrockAgentCoreApp with the invoke entrypoint."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp  # type: ignore[import-untyped]

    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def invoke(payload):
        """ACR invocation handler — fire background task, return immediately.

        Accepts two payload formats:

        **Training payload** (from ``AgentCoreOrchestrator``)::

            {"execution_id": "...", "proxy_url": "...", "task": {...}, "agent_config": {...}}

        **Test payload** (from ``agentcore invoke``)::

            {"prompt": "Hello"}

        Test payloads return a health-check response without running the agent.
        """
        # Parse the payload (JSON string from inputText)
        if isinstance(payload, str):
            payload = json.loads(payload)

        # Simple test / health-check invocation (e.g. `agentcore invoke '{"prompt": "Hi"}'`)
        if "execution_id" not in payload:
            agent_name = _agent_func.__module__ + "." + _agent_func.__name__ if _agent_func else "not loaded"
            return {
                "status": "ok",
                "agent": agent_name,
                "message": "rLLM AgentCore worker is running. "
                "Send a training payload with execution_id, proxy_url, task, and agent_config to run a rollout.",
            }

        execution_id = payload["execution_id"]
        proxy_url = payload["proxy_url"]
        task = payload["task"]
        agent_config = payload.get("agent_config", {})

        logger.info("Received invocation for execution_id=%s", execution_id)

        # Fire background task and return immediately
        task_id = app.add_async_task("rollout")
        asyncio.create_task(
            _run_and_push(execution_id, proxy_url, task, agent_config, task_id)
        )

        return {"status": "accepted", "execution_id": execution_id}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentCore worker for rLLM sandboxed execution"
    )
    parser.add_argument(
        "--agent-module",
        default=os.environ.get("RLLM_AGENT_MODULE", "agent"),
        help="Python module containing the agent function "
        "(default: $RLLM_AGENT_MODULE or 'agent')",
    )
    parser.add_argument(
        "--agent-func",
        default=os.environ.get("RLLM_AGENT_FUNC", "rollout"),
        help="Function name in the agent module "
        "(default: $RLLM_AGENT_FUNC or 'rollout')",
    )
    parser.add_argument(
        "--agent-dir",
        default=os.environ.get("RLLM_AGENT_DIR"),
        help="Directory to add to sys.path for agent import "
        "(default: $RLLM_AGENT_DIR)",
    )
    return parser.parse_args()


def main() -> None:
    global _agent_func, _app

    args = _parse_args()
    _agent_func = _load_agent_func(args.agent_module, args.agent_func, args.agent_dir)
    logger.info("Loaded agent function: %s.%s", args.agent_module, args.agent_func)

    _app = _create_app()
    _app.run()


if __name__ == "__main__":
    main()
