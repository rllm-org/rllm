from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rllm.sandbox.backends.modal_backend import ModalSandbox
from rllm.sandbox.network_policy import agent_network_policy


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("RLLM_AGENT_NETWORK_DOMAINS", raising=False)
    monkeypatch.delenv("RLLM_AGENT_NETWORK_CIDRS", raising=False)


def test_disabled():
    assert agent_network_policy("unused") is None


def test_gateway_ip(monkeypatch):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", '["pypi.org"]')
    assert agent_network_policy("http://5.78.144.17:19090/sessions/test/v1") == {
        "outbound_domain_allowlist": ["pypi.org"],
        "outbound_cidr_allowlist": ["5.78.144.17/32"],
    }


def test_empty_allows_only_gateway(monkeypatch):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", "[]")
    assert agent_network_policy("https://gateway.example/v1") == {
        "outbound_domain_allowlist": ["gateway.example"],
        "outbound_cidr_allowlist": [],
    }


@pytest.mark.parametrize("raw", ['"pypi.org"', '[1]', '[""]', 'not-json'])
def test_bad_config(monkeypatch, raw):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", raw)
    with pytest.raises(ValueError):
        agent_network_policy("https://gateway.example")


def test_http_hostname_rejected(monkeypatch):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", "[]")
    with pytest.raises(ValueError, match="HTTPS/443"):
        agent_network_policy("http://gateway.example:19090")


@pytest.mark.parametrize("fails", [False, True])
def test_restores_after_agent(monkeypatch, fails):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", "[]")
    update = Mock()
    sb = object.__new__(ModalSandbox)
    sb._agent_network_enabled = True
    sb._sandbox = SimpleNamespace(_experimental_set_outbound_network_policy=update)
    try:
        with sb.agent_network("https://gateway.example"):
            assert update.call_count == 1
            assert update.call_args.kwargs["outbound_cidr_allowlist"] == []
            if fails:
                raise RuntimeError("agent failed")
    except RuntimeError:
        assert fails
    assert update.call_count == 2
    assert update.call_args.kwargs["outbound_domain_allowlist"] == ["*"]


def test_policy_failure_never_runs_agent(monkeypatch):
    monkeypatch.setenv("RLLM_AGENT_NETWORK_DOMAINS", "[]")
    sb = object.__new__(ModalSandbox)
    sb._agent_network_enabled = True
    sb._sandbox = SimpleNamespace(_experimental_set_outbound_network_policy=Mock(side_effect=RuntimeError("denied")))
    entered = False
    with pytest.raises(RuntimeError, match="denied"):
        with sb.agent_network("https://gateway.example"):
            entered = True
    assert not entered
