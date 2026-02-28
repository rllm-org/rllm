"""Client helper for submitting execution results to the proxy."""

from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)


async def submit_result(proxy_base_url: str, execution_id: str, result_data: dict) -> None:
    """POST an execution result to the proxy result store route.

    Args:
        proxy_base_url: The proxy base URL (e.g. ``http://127.0.0.1:4000/v1``).
            The ``/v1`` suffix is stripped automatically.
        execution_id: Unique execution identifier.
        result_data: Serialized ExecutionResult dict.
    """
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
