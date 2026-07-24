from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import rllm.sandbox.backends.docker as docker_mod
from rllm.sandbox.backends.docker import DockerSandbox
from rllm.sandbox.protocol import SandboxCommandTimeout


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


def _sandbox_with_exec_result(exit_code: int = 0, output=(b"ok", b"")):
    calls = []

    def exec_run(*args, **kwargs):
        calls.append((args, kwargs))
        return exit_code, output

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.name = "task-1"
    sandbox._container = SimpleNamespace(exec_run=exec_run)
    return sandbox, calls


def test_docker_exec_wraps_command_with_wall_timeout():
    sandbox, calls = _sandbox_with_exec_result()

    assert sandbox.exec("sleep 300; echo done", timeout=600, user="agent") == "ok"

    args, kwargs = calls[0]
    assert args[0] == [
        "timeout",
        "-s",
        "TERM",
        "-k",
        "10s",
        "600s",
        "bash",
        "-c",
        "sleep 300; echo done",
    ]
    assert kwargs == {"demux": True, "user": "agent"}


def test_docker_exec_timeout_exit_at_wall_raises_typed_timeout(monkeypatch):
    sandbox, _ = _sandbox_with_exec_result(exit_code=124)
    ticks = iter([0.0, 600.0])
    monkeypatch.setattr(docker_mod.time, "monotonic", lambda: next(ticks))

    with pytest.raises(SandboxCommandTimeout, match="timed out after 600s"):
        sandbox.exec("sleep forever", timeout=600)


def test_docker_exec_fast_exit_124_stays_runtime_error(monkeypatch):
    sandbox, _ = _sandbox_with_exec_result(exit_code=124)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(docker_mod.time, "monotonic", lambda: next(ticks))

    with pytest.raises(RuntimeError) as exc_info:
        sandbox.exec("exit 124", timeout=600)

    assert not isinstance(exc_info.value, SandboxCommandTimeout)
