"""SandboxedAgentFlow contract: configure() override routing."""

from __future__ import annotations

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow


class _Flow(SandboxedAgentFlow):
    def run(self, task, config, *, env):
        return None


def test_configure_applies_known_overrides_and_returns_leftovers():
    flow = _Flow()
    # no overrides → defaults untouched, nothing left over.
    assert flow.configure({}) == {}
    assert flow.sandbox_backend == "docker"
    assert flow.max_concurrent == 64  # class default (raised from 4 upstream)

    leftovers = flow.configure({"sandbox_backend": "daytona", "sandbox_concurrency": 2, "made_up_flag": 1})
    assert flow.sandbox_backend == "daytona"
    assert flow.max_concurrent == 2
    assert leftovers == {"made_up_flag": 1}
