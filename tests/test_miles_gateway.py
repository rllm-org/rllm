"""Gateway wiring for the AgentFlow path on the miles backend.

AgentFlow agents talk OpenAI over rLLM's gateway rather than going through MilesEngine's
token-in/token-out path, so the gateway has to know how to reach Miles' inference
servers. It dispatches on the engine class name and previously fell through to
"Unknown engine type ... no workers registered" -- a warning, then a gateway with no
backend and every agent call failing.
"""

import inspect

from rllm.gateway.manager import GatewayManager


class _FakeEngine:
    def __init__(self, addresses):
        self.server_addresses = addresses


class TestMilesEngineIsRouted:
    def test_dispatch_covers_miles_engine(self):
        src = inspect.getsource(GatewayManager.start)
        assert '"MilesEngine"' in src, "gateway would register zero workers for MilesEngine"

    def test_router_url_becomes_a_worker_url(self):
        urls = GatewayManager._http_worker_urls(None, _FakeEngine(["http://10.0.0.1:15000"]))
        assert urls == ["http://10.0.0.1:15000"]

    def test_bare_host_port_gets_a_scheme(self):
        # verl hands over host:port; miles hands over a full URL. Both must work.
        urls = GatewayManager._http_worker_urls(None, _FakeEngine(["10.0.0.1:15000"]))
        assert urls == ["http://10.0.0.1:15000"]

    def test_several_addresses_all_register(self):
        urls = GatewayManager._http_worker_urls(None, _FakeEngine(["a:1", "http://b:2"]))
        assert urls == ["http://a:1", "http://b:2"]


class TestMilesEngineExposesAddresses:
    def test_engine_publishes_its_router(self):
        # Checked by source, since constructing MilesEngine needs a tokenizer.
        from rllm.engine.rollout import miles_engine

        assert "self.server_addresses = [self.router_url]" in inspect.getsource(miles_engine.MilesEngine.__init__)
