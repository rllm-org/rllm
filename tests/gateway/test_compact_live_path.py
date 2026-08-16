"""The compact stack must be reachable end-to-end (audit: 'no live path').

Covers the three wiring points the audit found dead: the manager's store
allowlist, the RLLM_GATEWAY_STORE env override, and the trace-fetch format
selection that follows the store — the single trigger for the pipeline.
"""

import pytest
from omegaconf import OmegaConf

from rllm.gateway.manager import GatewayManager


def _mgr(monkeypatch, store=None, env_store=None):
    monkeypatch.delenv("RLLM_GATEWAY_STORE", raising=False)
    if env_store is not None:
        monkeypatch.setenv("RLLM_GATEWAY_STORE", env_store)
    gateway = {} if store is None else {"store": store}
    return GatewayManager(config=OmegaConf.create({"rllm": {"gateway": gateway}}), mode="thread")


def test_memory_compact_is_an_allowed_store(monkeypatch):
    assert _mgr(monkeypatch, store="compact").store == "compact"


def test_env_var_opts_into_compact_store(monkeypatch):
    assert _mgr(monkeypatch, env_store="compact").store == "compact"


def test_unknown_store_still_rejected(monkeypatch):
    with pytest.raises(ValueError):
        _mgr(monkeypatch, store="redis")


def test_trace_format_follows_store(monkeypatch):
    assert _mgr(monkeypatch, store="compact")._trace_format == "compact"
    assert _mgr(monkeypatch, store="memory")._trace_format is None


def test_no_env_means_legacy_everywhere(monkeypatch):
    """The contract: RLLM_GATEWAY_STORE is the ONLY trigger. Unset it and the
    manager selects the default store and the legacy fetch format."""
    mgr = _mgr(monkeypatch)
    assert mgr.store == "memory"
    assert mgr._trace_format is None


def test_only_the_exact_compact_value_overrides_config(monkeypatch):
    """RLLM_GATEWAY_STORE is the compact opt-in ONLY: any other value must
    not alter legacy store selection (review: env 'sqlite' silently switched
    a configured 'memory' run)."""
    assert _mgr(monkeypatch, store="memory", env_store="sqlite").store == "memory"
    assert _mgr(monkeypatch, store="compact", env_store="memory").store == "compact"
    assert _mgr(monkeypatch, store="memory", env_store="compact").store == "compact"


def test_manager_construction_never_mutates_environment(monkeypatch):
    import os

    monkeypatch.delenv("RLLM_GATEWAY_STORE", raising=False)
    _mgr(monkeypatch, store="compact")
    assert "RLLM_GATEWAY_STORE" not in os.environ
