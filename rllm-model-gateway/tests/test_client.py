

class TestConnectIsBounded:
    """The trace API is called with timeout=600s. A scalar httpx timeout also governs
    connect and pool acquisition, so a stalled connect hung the rollout for ten minutes
    with the gateway idle -- observed as a run stuck on its last few trajectories.
    """

    def test_connect_and_pool_do_not_inherit_the_read_timeout(self):
        from rllm_model_gateway.client import AsyncGatewayClient, GatewayClient

        for cls in (GatewayClient, AsyncGatewayClient):
            t = cls("http://127.0.0.1:1", timeout=600.0)._http._timeout
            assert t.read == 600.0, cls.__name__
            assert t.connect == 10.0, f"{cls.__name__}: connect must not inherit 600s"
            assert t.pool == 60.0, f"{cls.__name__}: pool must not inherit 600s"
