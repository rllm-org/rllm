"""The gateway must not let an agent disable its own trace capture.

Injection used to be `if "logprobs" not in payload`, so an agent sending
`logprobs: false` kept that value and the resulting trace had no per-token logprobs --
silently, and on every backend.
"""

from rllm_model_gateway.middleware import SessionRoutingMiddleware


def _mw(**kw):
    return SessionRoutingMiddleware(app=None, **kw)


class TestCaptureFlagsCannotBeDisabledByTheClient:
    def test_absent_logprobs_is_injected(self):
        p = {}
        _mw()._mutate(p)
        assert p["logprobs"] is True

    def test_client_false_is_overridden(self):
        p = {"logprobs": False}
        _mw()._mutate(p)
        assert p["logprobs"] is True, "an agent could otherwise destroy its own training data"

    def test_client_zero_is_overridden(self):
        p = {"logprobs": 0}
        _mw()._mutate(p)
        assert p["logprobs"] is True

    def test_client_asking_for_more_is_left_alone(self):
        # completions-style int: honour it rather than downgrading to True
        p = {"logprobs": 5}
        _mw()._mutate(p)
        assert p["logprobs"] == 5

    def test_return_token_ids_gets_the_same_treatment(self):
        p = {"return_token_ids": False}
        _mw()._mutate(p)
        assert p["return_token_ids"] is True

    def test_injection_can_still_be_turned_off_wholesale(self):
        p = {"logprobs": False}
        _mw(add_logprobs=False, add_return_token_ids=False)._mutate(p)
        assert p["logprobs"] is False and "return_token_ids" not in p


class TestWorkerFlavorCrossesProcessBoundary:
    """The subprocess gateway defaults to worker_flavor="vllm", so the flavor has to be
    on its command line. It was not, which meant an SGLang run injected vLLM's
    return_token_ids, captured no token IDs, and only failed later in Step validation
    with "length mismatch between response_ids and logprobs, got 0, N".
    """

    def _manager(self, engine_cls_name):
        from rllm.gateway.manager import GatewayManager

        gw = GatewayManager.__new__(GatewayManager)
        gw.port = 9451
        gw.store = "memory"
        gw.db_path = None
        gw.model = "Qwen/Qwen3-1.7B"
        gw.cumulative_token_mode = False
        gw.renderer_family = "auto"
        gw.worker_flavor = "sglang" if engine_cls_name == "MilesEngine" else "vllm"
        return gw

    def test_sglang_flavor_is_on_the_subprocess_command(self):
        cmd = self._manager("MilesEngine")._gateway_cmd(9451)
        assert "--worker-flavor" in cmd
        assert cmd[cmd.index("--worker-flavor") + 1] == "sglang"

    def test_vllm_flavor_is_on_the_subprocess_command(self):
        cmd = self._manager("VerlEngine")._gateway_cmd(9451)
        assert cmd[cmd.index("--worker-flavor") + 1] == "vllm"

    def test_cli_parses_worker_flavor_into_the_config(self):
        """Closes the loop: the flag manager emits must survive arg parsing."""
        import argparse

        from rllm_model_gateway.server import _load_config

        args = argparse.Namespace(
            config=None, host=None, port=None, db_path=None, log_level=None,
            store=None, model=None, worker_flavor="sglang",
            cumulative_token_mode=False, renderer_family=None,
        )
        assert _load_config(args).worker_flavor == "sglang"


class TestSglangFlagsMatchMiles:
    def test_no_stop_trim_not_skip_special_tokens(self):
        """miles.rollout.session.core sets no_stop_trim=False and never touches
        skip_special_tokens; sending the latter left special tokens in the agent's text.
        """
        from rllm_model_gateway.middleware import SessionRoutingMiddleware

        mw = SessionRoutingMiddleware.__new__(SessionRoutingMiddleware)
        mw.add_logprobs = True
        mw.add_return_token_ids = True
        mw.worker_flavor = "sglang"
        mw.model = None
        mw.sessions = None

        payload = {"messages": []}
        mw._mutate(payload)
        assert payload["return_meta_info"] is True
        assert payload["return_prompt_token_ids"] is True
        assert payload["no_stop_trim"] is False
        assert "skip_special_tokens" not in payload
        assert "return_token_ids" not in payload
