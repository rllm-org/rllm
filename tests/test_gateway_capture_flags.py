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
