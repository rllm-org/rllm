"""The compact stack must be reachable end-to-end (audit: 'no live path').

Covers the three wiring points the audit found dead: the manager's store
allowlist, the RLLM_GATEWAY_STORE env override, and the trace-fetch format
selection that follows the store (or RLLM_COMPACT_TRACES).
"""

import pytest
from omegaconf import OmegaConf

from rllm.gateway.manager import GatewayManager


def _mgr(monkeypatch, store=None, env_store=None, env_compact=None):
    monkeypatch.delenv("RLLM_GATEWAY_STORE", raising=False)
    monkeypatch.delenv("RLLM_COMPACT_TRACES", raising=False)
    if env_store is not None:
        monkeypatch.setenv("RLLM_GATEWAY_STORE", env_store)
    if env_compact is not None:
        monkeypatch.setenv("RLLM_COMPACT_TRACES", env_compact)
    gateway = {} if store is None else {"store": store}
    return GatewayManager(config=OmegaConf.create({"rllm": {"gateway": gateway}}), mode="thread")


def test_memory_compact_is_an_allowed_store(monkeypatch):
    assert _mgr(monkeypatch, store="memory-compact").store == "memory-compact"


def test_env_var_opts_into_compact_store(monkeypatch):
    assert _mgr(monkeypatch, env_store="memory-compact").store == "memory-compact"


def test_unknown_store_still_rejected(monkeypatch):
    with pytest.raises(ValueError):
        _mgr(monkeypatch, store="redis")


def test_trace_format_follows_store(monkeypatch):
    assert _mgr(monkeypatch, store="memory-compact")._trace_format == "compact"
    assert _mgr(monkeypatch, store="memory")._trace_format is None


def test_trace_format_env_override(monkeypatch):
    assert _mgr(monkeypatch, store="memory", env_compact="1")._trace_format == "compact"
    assert _mgr(monkeypatch, store="memory-compact", env_compact="0")._trace_format is None
