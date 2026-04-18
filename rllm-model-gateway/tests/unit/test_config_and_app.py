"""GatewayConfig + create_app validation, lifecycle, and CLI config loading."""

from __future__ import annotations

import argparse

import pytest
from rllm_model_gateway import (
    GatewayConfig,
    NormalizedRequest,
    NormalizedResponse,
    create_app,
)
from rllm_model_gateway.server import _load_config
from rllm_model_gateway.store.memory_store import MemoryTraceStore


async def _adapter(_: NormalizedRequest) -> NormalizedResponse:
    return NormalizedResponse(content="ok", finish_reason="stop")


def test_create_app_requires_adapter_or_upstream():
    with pytest.raises(ValueError, match="adapter.*upstream_url"):
        create_app(GatewayConfig(), store=MemoryTraceStore())


def test_create_app_warns_when_both_adapter_and_upstream(caplog):
    cfg = GatewayConfig(upstream_url="http://nowhere")
    with caplog.at_level("WARNING"):
        create_app(cfg, adapter=_adapter, store=MemoryTraceStore())
    assert any("adapter takes precedence" in r.message for r in caplog.records)


def test_create_app_auto_generates_keys():
    cfg = GatewayConfig()
    create_app(cfg, adapter=_adapter, store=MemoryTraceStore())
    assert cfg.admin_api_key and len(cfg.admin_api_key) >= 32
    assert cfg.agent_api_key and len(cfg.agent_api_key) >= 32
    assert cfg.admin_api_key != cfg.agent_api_key


def test_create_app_preserves_provided_keys():
    cfg = GatewayConfig(admin_api_key="my-admin", agent_api_key="my-agent")
    create_app(cfg, adapter=_adapter, store=MemoryTraceStore())
    assert cfg.admin_api_key == "my-admin"
    assert cfg.agent_api_key == "my-agent"


def _empty_args(**overrides):
    base = dict(
        host=None,
        port=None,
        config=None,
        db_path=None,
        log_level=None,
        upstream_url=None,
        upstream_key=None,
        model=None,
        sampling_params_priority=None,
        admin_key=None,
        agent_key=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_load_config_env_var_precedence(monkeypatch):
    monkeypatch.setenv("RLLM_GATEWAY_PORT", "12345")
    monkeypatch.setenv("RLLM_GATEWAY_DB_PATH", "/tmp/x.db")
    monkeypatch.setenv("RLLM_GATEWAY_ADMIN_KEY", "from-env")
    monkeypatch.setenv("RLLM_GATEWAY_UPSTREAM_KEY", "upstream-from-env")
    cfg = _load_config(_empty_args())
    assert cfg.port == 12345
    assert cfg.db_path == "/tmp/x.db"
    assert cfg.admin_api_key == "from-env"
    assert cfg.upstream_api_key == "upstream-from-env"


def test_load_config_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("RLLM_GATEWAY_PORT", "9000")
    cfg = _load_config(
        _empty_args(
            host="localhost",
            port=8000,
            upstream_url="http://up",
            upstream_key="cli-upstream",
            model="m",
            sampling_params_priority="session",
            admin_key="cli-admin",
            agent_key="cli-agent",
        )
    )
    assert cfg.port == 8000
    assert cfg.upstream_url == "http://up"
    assert cfg.upstream_api_key == "cli-upstream"
    assert cfg.model == "m"
    assert cfg.sampling_params_priority == "session"
    assert cfg.admin_api_key == "cli-admin"
    assert cfg.agent_api_key == "cli-agent"


def test_load_config_yaml(tmp_path):
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text("port: 7777\nupstream_url: http://yaml-upstream\nupstream_api_key: yaml-key\nmodel: yaml-model\nsampling_params_priority: session\n")
    cfg = _load_config(_empty_args(config=str(yaml_path)))
    assert cfg.port == 7777
    assert cfg.upstream_url == "http://yaml-upstream"
    assert cfg.upstream_api_key == "yaml-key"
    assert cfg.model == "yaml-model"
    assert cfg.sampling_params_priority == "session"
