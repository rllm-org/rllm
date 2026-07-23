from __future__ import annotations

import sys
from types import SimpleNamespace

from rllm.sandbox.backends.docker import DockerSandbox


def test_docker_sandbox_maps_host_gateway(monkeypatch):
    calls = []
    client_options = []
    container = SimpleNamespace(short_id="abc123")
    client = SimpleNamespace(containers=SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)) or container))
    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(from_env=lambda **kwargs: client_options.append(kwargs) or client),
    )

    DockerSandbox(name="gateway-test", image="task-image")

    assert client_options == [{"timeout": 300.0}]
    _, kwargs = calls[0]
    assert kwargs["extra_hosts"] == {
        "host.docker.internal": "host-gateway",
    }


def test_docker_sandbox_honors_client_timeout(monkeypatch):
    client_options = []
    container = SimpleNamespace(short_id="abc123")
    client = SimpleNamespace(
        containers=SimpleNamespace(
            run=lambda *args, **kwargs: container,
        )
    )
    monkeypatch.setenv("RLLM_DOCKER_CLIENT_TIMEOUT_S", "420")
    monkeypatch.setitem(
        sys.modules,
        "docker",
        SimpleNamespace(from_env=lambda **kwargs: client_options.append(kwargs) or client),
    )

    DockerSandbox(name="timeout-test", image="task-image")

    assert client_options == [{"timeout": 420.0}]
