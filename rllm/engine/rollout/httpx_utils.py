"""Shared HTTPX lifecycle fixes for rollout backends."""

import logging

from httpx._client import BoundAsyncStream

logger = logging.getLogger(__name__)


def install_httpx_response_cycle_patch() -> None:
    """Release closed async responses without waiting for cyclic GC.

    HTTPX 0.28's ``BoundAsyncStream`` owns its ``Response`` while the response
    owns the stream. ``aclose()`` closes the transport but leaves that
    back-reference intact, so the response body remains reachable until cyclic
    GC runs. Both Tinker and Fireworks sampling use HTTPX; large Fireworks
    routing-matrix responses make the retained memory especially visible.

    Once ``aclose`` has finished, HTTPX has already recorded ``elapsed`` and
    released the connection, so the back-reference has no remaining purpose.
    ``BoundAsyncStream`` is internal to the lockfile's HTTPX version, so an
    incompatible future HTTPX change should fail loudly instead of silently
    restoring unbounded worker memory.
    """
    original_aclose = BoundAsyncStream.aclose
    if getattr(original_aclose, "_rllm_response_cycle_patch", False):
        return

    async def _aclose(self):  # noqa: ANN001 - matches HTTPX's private method
        try:
            return await original_aclose(self)
        finally:
            # BoundAsyncStream.aclose has already consumed this reference to set
            # Response.elapsed. Clearing it breaks Response <-> stream.
            if getattr(self, "_response", None) is not None:
                self._response = None

    _aclose._rllm_response_cycle_patch = True  # type: ignore[attr-defined]
    BoundAsyncStream.aclose = _aclose
    logger.info("Installed HTTPX closed-response cycle patch")
