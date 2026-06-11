"""Unit tests for SessionRoutingMiddleware._mutate.

Configured session keys overwrite whatever the client sent; unconfigured keys
pass through untouched.
"""

from __future__ import annotations

from rllm_model_gateway.middleware import SessionRoutingMiddleware


class _FakeSessions:
    def __init__(self, sampling_params: dict | None) -> None:
        self._sp = sampling_params

    def get_sampling_params(self, session_id: str):  # noqa: ARG002
        return self._sp


def _mw(sampling_params=None, *, add_logprobs=False, add_return_token_ids=False, model=None):
    return SessionRoutingMiddleware(
        app=lambda *a, **k: None,
        add_logprobs=add_logprobs,
        add_return_token_ids=add_return_token_ids,
        sessions=_FakeSessions(sampling_params),
        model=model,
    )


def test_configured_key_overwrites_client_value():
    mw = _mw({"temperature": 0.2})
    payload = {"temperature": 0.9, "messages": []}
    mw._mutate(payload, session_id="s1")
    assert payload["temperature"] == 0.2  # gateway wins


def test_unconfigured_key_passes_through():
    mw = _mw({"temperature": 0.2})
    payload = {"temperature": 0.9, "top_p": 0.5}
    mw._mutate(payload, session_id="s1")
    assert payload["temperature"] == 0.2  # owned by config
    assert payload["top_p"] == 0.5  # not in config → flow keeps it


def test_extra_keys_injected():
    mw = _mw({"presence_penalty": 0.1, "min_p": 0.05})
    payload = {"messages": []}
    mw._mutate(payload, session_id="s1")
    assert payload["presence_penalty"] == 0.1
    assert payload["min_p"] == 0.05


def test_no_session_params_leaves_payload_untouched():
    mw = _mw(None)
    payload = {"temperature": 0.9}
    mw._mutate(payload, session_id="s1")
    assert payload == {"temperature": 0.9}


def test_no_session_id_skips_injection():
    mw = _mw({"temperature": 0.2})
    payload = {"temperature": 0.9}
    mw._mutate(payload, session_id=None)
    assert payload["temperature"] == 0.9


def test_model_pin_and_logprobs_still_applied():
    mw = _mw({"temperature": 0.2}, add_logprobs=True, add_return_token_ids=True, model="served-model")
    payload = {"model": "whatever", "messages": []}
    mw._mutate(payload, session_id="s1")
    assert payload["model"] == "served-model"
    assert payload["logprobs"] is True
    assert payload["return_token_ids"] is True
    assert payload["temperature"] == 0.2
