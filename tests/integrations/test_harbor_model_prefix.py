"""Model names handed to Harbor agents must be parseable by litellm.

The agent inside the sandbox calls the model through litellm, which reads the
text before the first "/" as the provider. A served name like
``nvidia/nemotron-3-ultra-550b-a55b`` fails before any request is made:

    litellm.BadRequestError: LLM Provider NOT provided. ...
    You passed model=nvidia/nemotron-3-ultra-550b-a55b

Wrapping as ``openai/<name>`` routes to LLM_BASE_URL (rLLM's proxy/tunnel)
with the full name on the wire — the exact name the proxy serves.
"""

from rllm.integrations.harbor.trial_helper import MODEL_PLACEHOLDER, _litellm_routable


def test_unknown_provider_prefix_is_wrapped():
    assert _litellm_routable("nvidia/nemotron-3-ultra-550b-a55b") == "openai/nvidia/nemotron-3-ultra-550b-a55b"


def test_known_provider_prefixes_are_untouched():
    for name in ("anthropic/claude-sonnet-4-6", "openai/gpt-4o", "openrouter/qwen/qwen3-coder", "hosted_vllm/my-model"):
        assert _litellm_routable(name) == name


def test_wrapping_is_idempotent():
    once = _litellm_routable("nvidia/nemotron-3-ultra-550b-a55b")
    assert _litellm_routable(once) == once


def test_bare_names_keep_the_existing_inference():
    assert _litellm_routable("claude-sonnet-4-6") == "anthropic/claude-sonnet-4-6"
    assert _litellm_routable("my-local-model") == "openai/my-local-model"


def test_training_placeholder_is_untouched():
    assert _litellm_routable(MODEL_PLACEHOLDER) == MODEL_PLACEHOLDER
