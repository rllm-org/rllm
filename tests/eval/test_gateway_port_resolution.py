"""Port resolution for an eval gateway sitting behind a tunnel.

Concurrent evals against *remote* sandboxes each need their own tunnel on
their own port. Daemon state and the setup config are both process-wide
singletons, so without an explicit override every run reads the same port and
the second one dies on bind. ``RLLM_GATEWAY_PORT`` is that override.
"""

from __future__ import annotations

import pytest

import rllm.eval.runner as runner_mod
from rllm.eval.runner import ENV_GATEWAY_PORT, resolve_gateway_port

URL = "https://gw.example.test"


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(ENV_GATEWAY_PORT, raising=False)


def _daemon(monkeypatch, *, url: str, port: int) -> None:
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setattr(tunnel_mod, "live_tunnel", lambda: {"backend": "ngrok", "url": url, "pid": 1, "upstream": f"http://127.0.0.1:{port}"})


def _config_port(monkeypatch, port: int) -> None:
    import rllm.eval.config as config_mod

    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"port": port})


def test_env_override_wins_over_daemon_state(monkeypatch):
    """The whole point: two runs sharing one daemon still get distinct ports."""
    _daemon(monkeypatch, url=URL, port=9091)
    monkeypatch.setenv(ENV_GATEWAY_PORT, "9092")

    assert resolve_gateway_port(URL) == 9092


def test_daemon_upstream_used_when_url_matches(monkeypatch):
    _daemon(monkeypatch, url=URL, port=9091)
    _config_port(monkeypatch, 4321)

    assert resolve_gateway_port(URL) == 9091


def test_daemon_ignored_when_url_differs(monkeypatch):
    """A daemon forwarding somewhere else says nothing about this tunnel."""
    _daemon(monkeypatch, url="https://stale.example.test", port=9091)
    _config_port(monkeypatch, 4321)

    assert resolve_gateway_port(URL) == 4321


def test_falls_back_to_config_and_warns(monkeypatch, caplog):
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setattr(tunnel_mod, "live_tunnel", lambda: None)
    _config_port(monkeypatch, 4321)

    with caplog.at_level("WARNING", logger=runner_mod.__name__):
        assert resolve_gateway_port(URL) == 4321

    assert ENV_GATEWAY_PORT in caplog.text


def test_resolved_port_reaches_the_gateway(monkeypatch):
    """The resolved port must actually be handed to the gateway.

    Resolution being correct is worthless if ``run_dataset`` drops the value on
    the way to ``EvalGatewayManager`` — the gateway would then pick a free port
    and the tunnel would forward to nothing. Guards the wiring against a future
    refactor, which unit-testing the resolver alone cannot do.
    """
    import asyncio

    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setenv(ENV_GATEWAY_PORT, "9137")
    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", lambda: (URL, None))
    monkeypatch.setattr(tunnel_mod, "is_local_sandbox_backend", lambda name: False)
    monkeypatch.setattr(runner_mod, "is_local_sandbox_backend", lambda name: False, raising=False)

    seen: dict = {}

    class _Gateway:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def start(self):
            raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(manager_mod, "EvalGatewayManager", _Gateway)
    monkeypatch.setattr(runner_mod, "EvalGatewayManager", _Gateway, raising=False)

    with pytest.raises(_Stop):
        asyncio.run(
            runner_mod.run_dataset(
                tasks=[],
                agent_flow=object(),
                base_url="http://127.0.0.1:1/v1",
                model="m",
                sandbox_backend="modal",
            )
        )

    assert seen["port"] == 9137
    assert seen["tunnel"] == URL
