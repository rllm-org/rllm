"""Memory-lifecycle regressions for the Fireworks rollout engine."""

import asyncio
import gc
import weakref

import httpx
import pytest


def test_closed_httpx_response_does_not_wait_for_cyclic_gc():
    pytest.importorskip("fireworks.training.sdk")
    pytest.importorskip("tinker")

    # Import installs the guarded patch used by Fireworks gateway workers.
    from rllm.engine.rollout import fireworks_engine  # noqa: F401

    class PayloadStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"routing-matrix-payload"

        async def aclose(self) -> None:
            pass

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=PayloadStream())

    async def exercise() -> None:
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                response = await client.get("https://example.test/inference/v1/completions")
                stream = response.stream
                response_ref = weakref.ref(response)

                assert response.is_closed
                assert getattr(stream, "_response", None) is None
                del response
                assert response_ref() is None
        finally:
            if gc_was_enabled:
                gc.enable()

    asyncio.run(exercise())
