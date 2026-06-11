"""SandboxedAgentFlow contract: stateless flows, configure() override routing."""

from __future__ import annotations

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow


class _Flow(SandboxedAgentFlow):
    def run(self, task, config, *, env):
        return None


def test_configure_applies_known_overrides_and_returns_leftovers():
    flow = _Flow()
    leftovers = flow.configure({"sandbox_backend": "daytona", "sandbox_concurrency": 2, "made_up_flag": 1})
    assert flow.sandbox_backend == "daytona"
    assert flow.max_concurrent == 2
    assert leftovers == {"made_up_flag": 1}


def test_configure_leaves_defaults_when_not_overridden():
    flow = _Flow()
    assert flow.configure({}) == {}
    assert flow.sandbox_backend == "docker"
    assert flow.max_concurrent == 4


def test_flow_holds_no_sandbox_state():
    flow = _Flow()
    for attr in ("_sandbox", "sandbox", "set_sandbox", "create_instance", "teardown_sandbox"):
        assert not hasattr(flow, attr), f"stateless flows must not have {attr}"
