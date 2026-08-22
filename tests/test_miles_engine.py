"""MilesEngine sampling params.

A missing top_k let SGLang fall back to the model's generation_config (top_k=20 for
Qwen3), so the returned logprobs were renormalised over 20 tokens and systematically too
high. exp(train - rollout) then sat near 0.25 instead of 1.0, and TIS multiplied the
gradient by that bogus ratio. Nothing surfaced without TIS, because the loss never read
rollout_log_probs.
"""

from omegaconf import OmegaConf

from rllm.engine.rollout.miles_engine import MilesEngine


class TestSamplingParams:
    def test_top_k_is_always_sent(self):
        assert MilesEngine._sampling_from({})["top_k"] == -1

    def test_top_k_defaults_to_disabled_not_omitted(self):
        # -1 disables truncation; omitting the key is what let the server default win.
        params = MilesEngine._sampling_from(OmegaConf.create({"temperature": 1.0}))
        assert params["top_k"] == -1

    def test_config_overrides_the_default(self):
        params = MilesEngine._sampling_from(OmegaConf.create({"top_k": 50, "temperature": 0.7}))
        assert params["top_k"] == 50
        assert params["temperature"] == 0.7

    def test_defaults_are_neutral(self):
        params = MilesEngine._sampling_from(None)
        assert params == {"temperature": 1.0, "top_p": 1.0, "top_k": -1}

    def test_none_values_do_not_clobber_defaults(self):
        params = MilesEngine._sampling_from(OmegaConf.create({"top_p": None}))
        assert params["top_p"] == 1.0

    def test_unknown_keys_are_dropped(self):
        params = MilesEngine._sampling_from(OmegaConf.create({"max_tokens": 256, "bogus": 1}))
        assert "max_tokens" not in params and "bogus" not in params

    def test_optional_knobs_pass_through(self):
        params = MilesEngine._sampling_from(OmegaConf.create({"min_p": 0.05, "repetition_penalty": 1.1}))
        assert params["min_p"] == 0.05 and params["repetition_penalty"] == 1.1
