"""LiteLLM proxy callbacks for eval-time provider quirks.

Registered by ``EvalProxyManager._generate_litellm_config`` via
``litellm_settings: callbacks`` in the generated proxy config.
"""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger

_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


class _StripThinkingBlocks(CustomLogger):
    """Drop reasoning/thinking blocks from assistant messages before the upstream call.

    Anthropic-protocol CLIs (claude-code) echo ``thinking`` blocks back on
    later turns; litellm's Anthropic-to-chat-completions bridge carries them
    onto the assistant message as ``thinking_blocks``. Strict
    OpenAI-compatible providers (Fireworks: "Extra inputs are not permitted,
    field: 'messages[n].thinking_blocks'") 400 on the unknown field, which
    kills the agent loop after its first tool call. Anthropic-native
    downstreams NEED the blocks preserved, so the proxy only registers this
    callback for providers that don't consume them.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                # OpenAI-shaped (already translated): thinking_blocks field.
                message.pop("thinking_blocks", None)
                # Anthropic-shaped (raw /v1/messages body): content blocks.
                content = message.get("content")
                if isinstance(content, list):
                    kept = [
                        block
                        for block in content
                        if not (isinstance(block, dict) and block.get("type") in _THINKING_BLOCK_TYPES)
                    ]
                    message["content"] = kept if kept else ""
        return data


strip_thinking_blocks = _StripThinkingBlocks()
