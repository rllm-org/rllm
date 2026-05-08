from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace


def test_remote_retry_patch_retries_transport_errors(monkeypatch):
    patch_mod = importlib.import_module("swe.patches.swerex_remote_retry")
    patch_mod = importlib.reload(patch_mod)
    assert patch_mod.apply_swerex_remote_retry_patch() is True

    from swerex.runtime.remote import RemoteRuntime

    call_count = 0

    class FakePayload:
        def model_dump(self):
            return {"command": "pwd"}

    class FakeOutput:
        def __init__(self, **kwargs):
            self.payload = kwargs

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {"ok": True}

    class FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise patch_mod.aiohttp.ClientConnectionError("connect failed")
            return FakeResponse()

    class FakeLogger:
        def __init__(self):
            self.warnings = []
            self.errors = []

        def warning(self, *args, **kwargs):
            self.warnings.append((args, kwargs))

        def error(self, *args, **kwargs):
            self.errors.append((args, kwargs))

    async def _noop_sleep(_delay):
        return None

    fake_runtime = SimpleNamespace(
        _api_url="https://runtime.test",
        _headers={},
        _config=SimpleNamespace(timeout=60),
        logger=FakeLogger(),
    )

    async def _handle_response_errors(_response):
        return None

    fake_runtime._handle_response_errors = _handle_response_errors

    monkeypatch.setattr(patch_mod.aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(patch_mod.asyncio, "sleep", _noop_sleep)

    result = asyncio.run(
        RemoteRuntime._request(
            fake_runtime,
            "execute",
            FakePayload(),
            FakeOutput,
        )
    )

    assert isinstance(result, FakeOutput)
    assert result.payload == {"ok": True}
    assert call_count == 2
    assert len(fake_runtime.logger.warnings) == 1
    assert fake_runtime.logger.errors == []


def test_is_retryable_transport_error_filters_generic_exceptions():
    patch_mod = importlib.import_module("swe.patches.swerex_remote_retry")
    patch_mod = importlib.reload(patch_mod)

    assert patch_mod._is_retryable_transport_error(ValueError("no")) is False
    assert patch_mod._is_retryable_transport_error(patch_mod.aiohttp.ClientConnectionError("yes")) is True


def test_remote_retry_patch_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("SWE_REX_REMOTE_RETRIES", "5")
    monkeypatch.setenv("SWE_REX_REMOTE_SOCK_CONNECT_TIMEOUT_S", "12.5")

    patch_mod = importlib.import_module("swe.patches.swerex_remote_retry")
    patch_mod = importlib.reload(patch_mod)

    assert patch_mod._DEFAULT_REMOTE_RETRIES == 5
    assert patch_mod._SOCK_CONNECT_TIMEOUT_S == 12.5
