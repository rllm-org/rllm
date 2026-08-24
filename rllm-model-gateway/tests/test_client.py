

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
            assert t.connect == 30.0, f"{cls.__name__}: connect must not inherit 600s"
            assert t.pool == 60.0, f"{cls.__name__}: pool must not inherit 600s"


    def test_flush_does_not_undo_the_bound_with_a_scalar(self):
        """flush() takes its own timeout arg. httpx replaces the client default outright
        when a request passes a *scalar*, so `timeout=600.0` there restored connect=600s
        and silently reopened the hang on the flush leg -- which is on the per-rollout
        path (aget_traces calls flush first).
        """
        from rllm_model_gateway.client import GatewayClient, _timeout

        http = GatewayClient("http://127.0.0.1:1", timeout=600.0)._http
        eff = http.build_request("POST", "http://x/admin/flush", timeout=_timeout(600.0)).extensions["timeout"]
        assert eff["connect"] == 30.0
        assert eff["pool"] == 60.0
        assert eff["read"] == 600.0, "the long read budget must survive"

        bare = http.build_request("POST", "http://x/admin/flush", timeout=600.0).extensions["timeout"]
        assert bare["connect"] == 600.0, "guard premise: a scalar really does override connect"
