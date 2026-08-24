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


class TestSGLangTraceExtraction:
    """SGLang has no vLLM-style choices[0].token_ids; completion ids ride in
    meta_info.output_token_logprobs as [logprob, token_id] pairs."""

    # what an SGLang chat response looks like with return_meta_info + logprobs
    SGLANG = {
        "choices": [
            {
                "message": {"content": "hi"},
                "prompt_token_ids": [1, 2, 3],
                "logprobs": {"content": [{"token": "h", "logprob": -0.5}, {"token": "i", "logprob": -0.25}]},
                "meta_info": {"output_token_logprobs": [[-0.5, 40, None], [-0.25, 41, None]]},
            }
        ]
    }
    VLLM = {
        "choices": [
            {
                "message": {"content": "hi"},
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [40, 41],
                "logprobs": {"content": [{"token": "h", "logprob": -0.5}, {"token": "i", "logprob": -0.25}]},
            }
        ]
    }

    def test_completion_ids_from_sglang_meta_info(self):
        from rllm_model_gateway.data_process import extract_completion_token_ids

        assert extract_completion_token_ids(self.SGLANG) == [40, 41]

    def test_vllm_token_ids_still_win(self):
        from rllm_model_gateway.data_process import extract_completion_token_ids

        assert extract_completion_token_ids(self.VLLM) == [40, 41]

    def test_prompt_ids_from_the_choice(self):
        from rllm_model_gateway.data_process import extract_prompt_token_ids

        assert extract_prompt_token_ids(self.SGLANG) == [1, 2, 3]

    def test_logprobs_come_from_meta_info_when_present(self):
        from rllm_model_gateway.data_process import extract_logprobs

        assert extract_logprobs(self.SGLANG) == [-0.5, -0.25]

    def test_ids_and_logprobs_are_index_aligned(self):
        from rllm_model_gateway.data_process import extract_completion_token_ids, extract_logprobs

        assert len(extract_completion_token_ids(self.SGLANG)) == len(extract_logprobs(self.SGLANG))

    def test_vllm_logprobs_path_unchanged(self):
        from rllm_model_gateway.data_process import extract_logprobs

        assert extract_logprobs(self.VLLM) == [-0.5, -0.25]

    def test_missing_meta_info_yields_nothing_rather_than_raising(self):
        from rllm_model_gateway.data_process import extract_completion_token_ids

        assert extract_completion_token_ids({"choices": [{"message": {}}]}) == []


class TestFlavourInjection:
    def test_sglang_gets_meta_info_flags(self):
        from rllm_model_gateway.middleware import SessionRoutingMiddleware

        p = {}
        SessionRoutingMiddleware(app=None, worker_flavor="sglang")._mutate(p)
        assert p["return_meta_info"] is True
        assert p["return_prompt_token_ids"] is True
        assert p["skip_special_tokens"] is False
        assert "return_token_ids" not in p, "SGLang has no such field"

    def test_vllm_keeps_return_token_ids(self):
        from rllm_model_gateway.middleware import SessionRoutingMiddleware

        p = {}
        SessionRoutingMiddleware(app=None, worker_flavor="vllm")._mutate(p)
        assert p["return_token_ids"] is True
        assert "return_meta_info" not in p

    def test_manager_sets_flavour_from_the_engine(self):
        import inspect

        from rllm.gateway.manager import GatewayManager

        src = inspect.getsource(GatewayManager.start)
        assert 'self.worker_flavor = "sglang" if engine_cls == "MilesEngine" else "vllm"' in src
