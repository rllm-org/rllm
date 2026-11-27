"""Health-probe configuration and real HTTP timeout regression tests."""

import argparse
import asyncio
from contextlib import suppress

import pytest
from pydantic import ValidationError
from rllm_model_gateway import GatewayConfig, create_app
from rllm_model_gateway.server import _load_config
from rllm_model_gateway.session_router import SessionRouter


@pytest.mark.asyncio
@pytest.mark.parametrize("settings,timeout,threshold", [({}, 5.0, 3), ({"health_check_timeout": 30.0, "health_failure_threshold": 8}, 30.0, 8)])
async def test_app_passes_health_settings_to_router(settings, timeout, threshold):
    app = create_app(GatewayConfig(**settings))
    router = app.state.router
    try:
        await router.start_health_checks()
        assert router._http.timeout.read == timeout
        assert router._http.timeout.connect == timeout
        assert router._failure_threshold == threshold
    finally:
        await router.stop_health_checks()
        await app.state.store.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("health_check_timeout", 0),
        ("health_check_timeout", -1),
        ("health_check_timeout", float("inf")),
        ("health_check_timeout", float("nan")),
        ("health_failure_threshold", 0),
        ("health_failure_threshold", -1),
        ("health_failure_threshold", 1.5),
    ],
)
def test_invalid_health_settings_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        GatewayConfig(**{field: value})


def test_health_settings_from_yaml_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("RLLM_GATEWAY_HEALTH_CHECK_TIMEOUT", raising=False)
    monkeypatch.delenv("RLLM_GATEWAY_HEALTH_FAILURE_THRESHOLD", raising=False)
    path = tmp_path / "gateway.yaml"
    path.write_text("health_check_timeout: 30.5\nhealth_failure_threshold: 6\n")
    args = argparse.Namespace(config=str(path))

    config = _load_config(args)
    assert config.health_check_timeout == 30.5
    assert config.health_failure_threshold == 6

    monkeypatch.setenv("RLLM_GATEWAY_HEALTH_CHECK_TIMEOUT", "60.5")
    monkeypatch.setenv("RLLM_GATEWAY_HEALTH_FAILURE_THRESHOLD", "9")
    config = _load_config(args)
    assert config.health_check_timeout == 60.5
    assert config.health_failure_threshold == 9


@pytest.mark.parametrize(
    "name,value,field",
    [
        ("RLLM_GATEWAY_HEALTH_CHECK_TIMEOUT", "invalid", "health_check_timeout"),
        ("RLLM_GATEWAY_HEALTH_FAILURE_THRESHOLD", "0", "health_failure_threshold"),
    ],
)
def test_invalid_health_environment_is_rejected(monkeypatch, name, value, field):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError, match=field):
        _load_config(argparse.Namespace())


@pytest.mark.asyncio
async def test_stalled_health_endpoint_times_out_and_recovers(monkeypatch):
    """Use a local HTTP worker so the real httpx timeout is exercised."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    ready = asyncio.Event()
    connections = set()

    async def health(reader, writer):
        task = asyncio.current_task()
        connections.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            await ready.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
            with suppress(ConnectionError):
                await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            connections.discard(task)

    server = await asyncio.start_server(health, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    app = create_app(GatewayConfig(health_check_timeout=0.1))
    router = app.state.router
    try:
        await router.start_health_checks()
        assert await asyncio.wait_for(router._check(url), timeout=2) == (url, False)
        ready.set()
        assert await asyncio.wait_for(router._check(url), timeout=2) == (url, True)
    finally:
        ready.set()
        await router.stop_health_checks()
        server.close()
        await server.wait_closed()
        if connections:
            await asyncio.gather(*connections)
        await app.state.store.close()


@pytest.mark.asyncio
async def test_router_positional_arguments_remain_compatible():
    router = SessionRouter(None, 10.0, 7)
    try:
        await router.start_health_checks()
        assert router._failure_threshold == 7
        assert router._http.timeout.read == 5.0
    finally:
        await router.stop_health_checks()
