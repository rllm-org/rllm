"""Eval-owned tunnel wiring."""

from __future__ import annotations

import asyncio

import pytest

import rllm.eval.runner as runner_mod


def test_remote_eval_skips_daemon_and_uses_automatic_port(monkeypatch):
    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod

    seen = {}

    def _resolve():
        return "cloudflared", None

    class _Stop(Exception):
        pass

    class _Gateway:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def start(self):
            raise _Stop

    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", _resolve)
    monkeypatch.setattr(tunnel_mod, "is_local_sandbox_backend", lambda _: False)
    monkeypatch.setattr(manager_mod, "EvalGatewayManager", _Gateway)

    with pytest.raises(_Stop):
        asyncio.run(
            runner_mod.run_dataset(
                tasks=[],
                agent_flow=object(),
                base_url="http://127.0.0.1:1/v1",
                model="probe-model",
                sandbox_backend="modal",
            )
        )

    assert seen["tunnel"] == "cloudflared"
    assert seen["port"] is None
