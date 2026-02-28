"""Client helper for submitting execution results to the proxy."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0


async def submit_result(
    proxy_base_url: str,
    execution_id: str,
    result_data: dict,
    session: aiohttp.ClientSession | None = None,
) -> None:
    """POST an execution result to the proxy result store route.

    Args:
        proxy_base_url: The proxy base URL (e.g. ``http://127.0.0.1:4000/v1``).
            The ``/v1`` suffix is stripped automatically.
        execution_id: Unique execution identifier.
        result_data: Serialized ExecutionResult dict.
        session: Optional shared session. A temporary one is created if not provided.
    """
    base = proxy_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    url = f"{base}/rllm/results/{execution_id}"

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        for attempt in range(_MAX_RETRIES):
            try:
                async with session.post(url, json=result_data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status < 400:
                        logger.debug("Result submitted for %s", execution_id)
                        return
                    body = await resp.text()
                    logger.error(
                        "Failed to submit result for %s (attempt %d/%d): HTTP %s – %s",
                        execution_id, attempt + 1, _MAX_RETRIES, resp.status, body,
                    )
            except Exception:
                logger.exception(
                    "Failed to submit result for %s (attempt %d/%d)",
                    execution_id, attempt + 1, _MAX_RETRIES,
                )

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))

        logger.error("Giving up submitting result for %s after %d attempts", execution_id, _MAX_RETRIES)
    finally:
        if own_session:
            await session.close()
