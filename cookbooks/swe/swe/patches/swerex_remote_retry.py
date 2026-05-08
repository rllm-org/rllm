"""Patch SWE-ReX remote runtime requests for transient Modal tunnel failures.

The upstream ``RemoteRuntime._request`` does not set an explicit connect timeout
and defaults to zero retries. On this machine we can intermittently get
``Cannot connect to host ...modal.host:443`` after a sandbox has already started.

SWE-ReX already sends an ``X-Request-ID`` idempotency key, so retrying
connection-level failures is safe. This patch keeps upstream behavior for
non-transport errors while adding:

1. A short socket-connect timeout, so dead tunnels fail fast.
2. A small retry budget for idempotent runtime RPCs.
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from typing import Any

import aiohttp

_patched = False


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


_DEFAULT_REMOTE_RETRIES = _read_nonnegative_int_env("SWE_REX_REMOTE_RETRIES", 3)
_SOCK_CONNECT_TIMEOUT_S = _read_positive_float_env("SWE_REX_REMOTE_SOCK_CONNECT_TIMEOUT_S", 15.0)
_RETRYABLE_ENDPOINTS = {
    "close_session",
    "create_session",
    "execute",
    "read_file",
    "run_in_session",
    "write_file",
}


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return True for connection setup errors worth retrying."""
    return isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
        ),
    )


def apply_swerex_remote_retry_patch() -> bool:
    """Add connect timeouts and retries to SWE-ReX remote runtime requests."""
    global _patched
    if _patched:
        return False

    from swerex.runtime.remote import RemoteRuntime

    async def _request_with_transport_retries(
        self: Any,
        endpoint: str,
        payload: Any,
        output_class: Any,
        num_retries: int = 0,
    ):
        request_url = f"{self._api_url}/{endpoint}"
        request_id = str(uuid.uuid4())
        headers = self._headers.copy()
        headers["X-Request-ID"] = request_id

        retry_budget = num_retries
        if retry_budget == 0 and endpoint in _RETRYABLE_ENDPOINTS:
            retry_budget = _DEFAULT_REMOTE_RETRIES

        timeout_total = getattr(self._config, "timeout", None)
        timeout = aiohttp.ClientTimeout(
            total=timeout_total if timeout_total and timeout_total > 0 else None,
            sock_connect=_SOCK_CONNECT_TIMEOUT_S,
        )

        attempt = 0
        retry_delay = 0.1
        backoff_max = 5.0

        while True:
            try:
                async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(force_close=True),
                    timeout=timeout,
                ) as session:
                    async with session.post(
                        request_url,
                        json=payload.model_dump() if payload else None,
                        headers=headers,
                    ) as resp:
                        await self._handle_response_errors(resp)
                        return output_class(**await resp.json())
            except Exception as exc:
                should_retry = attempt < retry_budget and _is_retryable_transport_error(exc)
                if should_retry:
                    attempt += 1
                    self.logger.warning(
                        "Transient remote runtime error on %s request %s (%s/%s): %s",
                        endpoint,
                        request_id,
                        attempt,
                        retry_budget,
                        exc,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2 + random.uniform(0, 0.5), backoff_max)
                    continue

                self.logger.error(
                    "Error making request %s after %d retries: %s",
                    request_id,
                    attempt,
                    exc,
                )
                raise

    RemoteRuntime._request = _request_with_transport_retries  # type: ignore[attr-defined]

    _patched = True
    return True
