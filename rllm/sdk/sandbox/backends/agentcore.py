"""AWS Bedrock AgentCore sandbox backend.

Bypasses the ``SandboxOrchestrator`` / ``Sandbox`` protocol entirely.  Instead
of uploading code into a managed sandbox and running ``worker_server.py``, we
invoke a pre-deployed AgentCore Runtime via ``boto3``.  The ACR container runs
``agentcore_worker.py`` which POSTs results directly to the proxy's
``/rllm/results/{execution_id}`` SQLite store — identical to local/docker.

Required config (via ``SandboxConfig.extra``)::

    extra:
      agent_runtime_arn: "arn:aws:bedrock-agentcore:us-west-2:123456:runtime/abc"
      base_url: "http://<host-ip>:4000/v1"  # reachable proxy URL for ACR container
      tps_limit: 25          # optional, default 25
      max_retry_attempts: 5  # optional, default 5
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from rllm.sdk.sandbox.protocol import ExecutionResult, SandboxConfig

logger = logging.getLogger(__name__)


def _extract_region(arn: str) -> str:
    """Extract the AWS region from an ARN string."""
    match = re.match(r"arn:aws[^:]*:([^:]+):([^:]+):", arn)
    if match:
        return match.group(2)
    raise ValueError(f"Cannot extract region from ARN: {arn!r}")


class AgentCoreOrchestrator:
    """Orchestrator that invokes AWS Bedrock AgentCore Runtimes.

    Conforms to the same ``initialize()`` / ``execute()`` / ``shutdown()``
    interface as ``SandboxOrchestrator`` so the engine layer can use it
    interchangeably.
    """

    def __init__(self, config: SandboxConfig):
        self._config = config

        # Validate required extra fields
        arn = config.extra.get("agent_runtime_arn")
        if not arn:
            raise ValueError(
                "AgentCore backend requires 'agent_runtime_arn' in sandbox.extra. "
                "Example: rllm.sdk.sandbox.extra.agent_runtime_arn=arn:aws:bedrock-agentcore:..."
            )
        self._arn: str = arn
        self._region: str = _extract_region(arn)

        self._tps_limit: int = int(config.extra.get("tps_limit", 25))
        self._max_retry: int = int(config.extra.get("max_retry_attempts", 5))

        self._client: Any = None
        self._semaphore: asyncio.Semaphore | None = None
        self._proxy_url: str = ""
        self._result_store: Any = None
        self._initialized = False

    async def initialize(self, proxy_url: str, result_store: Any) -> None:
        """Create the boto3 client and store references.

        ``boto3`` is imported lazily so the module can be imported without
        the AWS SDK installed.

        If ``config.extra["base_url"]`` is set it overrides *proxy_url*.
        This is required when the ACR container cannot reach the trainer's
        localhost (e.g. ``http://127.0.0.1:4000/v1``).  Set it to a
        publicly/VPC-reachable URL like ``http://<host-ip>:4000/v1``.
        """
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for the agentcore backend. "
                "Install it with: pip install boto3"
            ) from exc

        boto_config = BotoConfig(
            region_name=self._region,
            retries={"mode": "adaptive", "max_attempts": self._max_retry},
        )
        self._client = boto3.client("bedrock-agentcore", config=boto_config)

        # Allow overriding the proxy URL for remote containers that can't
        # reach the trainer's localhost address.
        override_url = self._config.extra.get("base_url")
        if override_url:
            self._proxy_url = override_url
            logger.info(
                "Using overridden proxy URL from config.extra.base_url: %s",
                override_url,
            )
        else:
            self._proxy_url = proxy_url
            logger.warning(
                "No base_url override in sandbox.extra — using proxy_url=%s. "
                "If the ACR container cannot reach this address, set "
                "rllm.sdk.sandbox.extra.base_url to a reachable URL.",
                proxy_url,
            )

        self._result_store = result_store
        self._semaphore = asyncio.Semaphore(self._tps_limit)
        self._initialized = True

        logger.info(
            "AgentCoreOrchestrator initialized (arn=%s, region=%s, tps_limit=%d)",
            self._arn,
            self._region,
            self._tps_limit,
        )

    async def execute(self, task: dict, agent_config: dict) -> ExecutionResult:
        """Invoke the AgentCore Runtime and wait for the result.

        1. Register execution_id in the result store.
        2. Invoke the ACR with the payload (execution_id, proxy_url, task, agent_config).
        3. Wait for the result via the result store.
        """
        if not self._initialized:
            raise RuntimeError("AgentCoreOrchestrator not initialized. Call initialize() first.")

        execution_id = str(uuid.uuid4())
        self._result_store.register(execution_id)

        # Build payload — same structure as worker_server.py's /execute body
        payload = {
            "execution_id": execution_id,
            "proxy_url": self._proxy_url,
            "task": task,
            "agent_config": agent_config,
        }

        assert self._semaphore is not None
        async with self._semaphore:
            try:
                await asyncio.to_thread(self._invoke_runtime, execution_id, payload)
            except Exception as exc:
                logger.error("Failed to invoke AgentCore runtime for %s: %s", execution_id, exc)
                return ExecutionResult(
                    success=False,
                    error=f"AgentCore invocation failed: {exc}",
                )

        # Wait for the result.  Use wait_async when available (falls back to
        # threaded polling for the cross-process proxy case).
        timeout = self._config.execution_timeout
        if hasattr(self._result_store, "wait_async"):
            return await self._result_store.wait_async(execution_id, timeout)
        return await asyncio.to_thread(
            self._result_store.wait,
            execution_id,
            timeout,
        )

    def _invoke_runtime(self, execution_id: str, payload: dict) -> None:
        """Synchronous boto3 call to invoke the AgentCore Runtime."""
        logger.debug("Invoking AgentCore runtime for %s", execution_id)
        response = self._client.invoke_agent_runtime(
            agentRuntimeArn=self._arn,
            runtimeSessionId=execution_id,  # UUID (36 chars) satisfies 33-char minimum
            payload=json.dumps(payload),
            qualifier="DEFAULT",
        )
        # The ACR agent receives the payload via its /invocations endpoint
        # and returns immediately after spawning a background task.  The
        # actual result will be POSTed back to the proxy's result store.
        logger.debug(
            "AgentCore invocation accepted for %s (HTTP %s)",
            execution_id,
            response.get("ResponseMetadata", {}).get("HTTPStatusCode", "?"),
        )

    async def shutdown(self) -> None:
        """Release the boto3 client."""
        self._client = None
        if self._result_store is not None:
            self._result_store.close()
        self._initialized = False
        logger.info("AgentCoreOrchestrator shut down")
