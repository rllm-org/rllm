"""rLLM-native renderer layer.

A single token-level renderer interface for training, with a registry that routes
each model to the strongest available backend:

- **prime-rl** (``renderers``) — token-level with a real cross-turn
  ``bridge_to_next_turn``; preferred for RL multi-turn rollouts.
- **tinker-cookbook / Fireworks** (``training.renderer``) — broader model
  coverage (DeepSeek-V4-Flash, Gemma-4, Ministral-3, Kimi-K2.7-code, …), adapted
  to the same interface.

Typical use::

    from rllm.renderers import resolve

    renderer = resolve("Qwen/Qwen3-8B", tokenizer)            # -> prime-rl (bridge)
    renderer = resolve("deepseek-ai/DeepSeek-V4-Flash", tok)  # -> Fireworks adapter
    ids = renderer.render_ids(messages, add_generation_prompt=True)
    parsed = renderer.parse_response(completion_ids)
    nxt = renderer.bridge_to_next_turn(prev_prompt, prev_completion, new_msgs)
"""

from __future__ import annotations

from rllm.renderers._fw_register import FIREWORKS_AVAILABLE
from rllm.renderers._prime import PRIME_AVAILABLE
from rllm.renderers._tinker import TINKER_AVAILABLE
from rllm.renderers.registry import (
    Backend,
    RendererResolution,
    describe,
    resolve,
    select_backend,
)
from rllm.renderers.types import (
    Message,
    ParsedResponse,
    RenderedTokens,
    Renderer,
    ToolCall,
    ToolSpec,
)

__all__ = [
    # Types / protocol
    "Renderer",
    "RenderedTokens",
    "ParsedResponse",
    "ToolCall",
    "ToolSpec",
    "Message",
    # Registry
    "resolve",
    "select_backend",
    "describe",
    "Backend",
    "RendererResolution",
    # Backend availability
    "PRIME_AVAILABLE",
    "TINKER_AVAILABLE",
    "FIREWORKS_AVAILABLE",
]
