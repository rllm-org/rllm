"""tinker-cookbook / Fireworks backend, adapted to the native :class:`Renderer`.

The tinker-cookbook ``Renderer`` (and the Fireworks cookbook subclasses layered
on top via ``training.renderer``) is a *message-level* abstraction. This adapter
exposes it through the native token-level contract so models prime-rl lacks —
DeepSeek-V4-Flash, Gemma-4, Ministral-3, Kimi-K2.7-code — are reachable for
training, **including a working cross-turn ``bridge_to_next_turn``**.

The bridge is generic: ``build_generation_prompt`` is itself
``bos + Σ render_message(m) + _get_generation_suffix()``, so the next-turn prompt
is the prior turn's *sampled* tokens (kept verbatim, anchored at the turn-close
token) followed by ``render_message`` for each new non-assistant message plus the
assistant opener. This preserves historical thinking the model actually emitted,
which a naive full re-render would strip. Validated byte-for-byte against
prime-rl's hand-coded bridge for the Qwen3 / Qwen3.5 families.

It lives here (in ``rllm``, where tinker-cookbook is already a dependency) rather
than in the model gateway, which stays free of tinker/Fireworks deps — the
gateway receives a built renderer by injection (in-process mode) and only ever
calls ``bridge_to_next_turn`` / reads ``.token_ids``.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from rllm.renderers._common import iter_tool_specs, normalize_tool_calls
from rllm.renderers.types import Message, ParsedResponse, RenderedTokens, ToolSpec

logger = logging.getLogger(__name__)

TINKER_AVAILABLE: bool = importlib.util.find_spec("tinker_cookbook") is not None
# The bridge is native (built on tinker primitives), so it's available whenever
# tinker-cookbook is.
BRIDGE_AVAILABLE: bool = TINKER_AVAILABLE


# HF model id -> registered renderer name, for models the upstream
# ``model_info.get_recommended_renderer_name`` does not know (Fireworks-cookbook
# additions that prime-rl also lacks). The registry consults prime-rl first, so
# only prime-rl gaps need to appear here. Extend as the cookbook adds models.
_FW_MODEL_RENDERER: dict[str, str] = {
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek_v4",
    "google/gemma-4-E2B-it": "gemma4",
    "mistralai/Ministral-3-3B-Instruct-2512": "mistral",
}


def tinker_renderer_name(model_name: str | None) -> str | None:
    """Resolve a tinker/Fireworks renderer name for ``model_name``, or ``None``.

    Checks the rLLM-side Fireworks override map first (because upstream
    ``model_info`` lags new models and does not know the cookbook additions),
    then falls back to ``model_info.get_recommended_renderer_name``.
    """
    if not TINKER_AVAILABLE or not model_name:
        return None
    if model_name in _FW_MODEL_RENDERER:
        return _FW_MODEL_RENDERER[model_name]
    try:
        from tinker_cookbook import model_info

        return model_info.get_recommended_renderer_name(model_name)
    except KeyError:
        return None
    except Exception as err:  # noqa: BLE001
        logger.debug("model_info lookup failed for %s: %s", model_name, err)
        return None


def _convert_messages(messages: list[Message]) -> list[Message]:
    """Convert OpenAI-style message dicts to tinker-cookbook Messages.

    Mirrors ``TinkerEngine._convert_openai_messages`` so rendered tokens match
    the existing engine path.
    """
    from tinker_cookbook.renderers.base import ToolCall as TinkerToolCall

    out: list[Message] = []
    for msg in messages:
        tinker_msg: Message = {"role": msg["role"], "content": msg.get("content") or ""}
        if "name" in msg:
            tinker_msg["name"] = msg["name"]
        if "tool_call_id" in msg:
            tinker_msg["tool_call_id"] = msg["tool_call_id"]
        if "tool_calls" in msg:
            tinker_msg["tool_calls"] = [
                TinkerToolCall.model_validate(tc) for tc in msg["tool_calls"]
            ]
        out.append(tinker_msg)
    return out


def _to_tinker_tool_specs(tools: list[ToolSpec] | list[dict[str, Any]] | None):
    specs = iter_tool_specs(tools)
    if not specs:
        return []
    from tinker_cookbook.renderers.base import ToolSpec as TinkerToolSpec

    return [
        TinkerToolSpec(name=s.name, description=s.description, parameters=s.parameters)
        for s in specs
    ]


class TinkerAdapter:
    """Native-protocol adapter over a tinker-cookbook / Fireworks renderer."""

    backend = "tinker"

    def __init__(self, renderer: Any, tokenizer: Any, renderer_name: str | None = None):
        self._renderer = renderer
        self._tokenizer = tokenizer
        self.name = renderer_name or type(renderer).__name__
        self._close_ids: list[int] | None = None
        self.has_bridge = True

    @property
    def has_extension_property(self) -> bool:
        return bool(getattr(self._renderer, "has_extension_property", False))

    def _prepare(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | list[dict[str, Any]] | None,
    ) -> list[Message]:
        tinker_messages = _convert_messages(messages)
        tool_specs = _to_tinker_tool_specs(tools)
        if not tool_specs:
            return tinker_messages
        # Fold tool declarations into a conversation prefix, pulling a leading
        # system message into the prefix's system slot (mirrors
        # TinkerEngine._build_messages_with_tools).
        system_prompt = ""
        if tinker_messages and tinker_messages[0]["role"] == "system":
            content = tinker_messages[0].get("content") or ""
            system_prompt = content if isinstance(content, str) else ""
            remaining = tinker_messages[1:]
        else:
            remaining = tinker_messages
        prefix = self._renderer.create_conversation_prefix_with_tools(
            tool_specs, system_prompt
        )
        return list(prefix) + list(remaining)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        prepared = self._prepare(messages, tools)
        if add_generation_prompt:
            model_input = self._renderer.build_generation_prompt(prepared)
        else:
            # Render the full conversation with no trailing generation prompt.
            # build_supervised_example returns (ModelInput, weights); the token
            # ids are independent of the loss mask. Requires a complete
            # conversation (ending in an assistant turn).
            model_input, _weights = self._renderer.build_supervised_example(prepared)
        return list(model_input.to_ints())

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        return RenderedTokens(
            token_ids=self.render_ids(
                messages, tools=tools, add_generation_prompt=add_generation_prompt
            )
        )

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        message, _termination = self._renderer.parse_response(token_ids)
        content = message["content"]
        reasoning = ""
        if isinstance(content, list):
            text_parts = [p for p in content if p.get("type") == "text"]
            think_parts = [p for p in content if p.get("type") == "thinking"]
            content = "\n".join(p["text"] for p in text_parts)
            reasoning = "\n".join(p["thinking"] for p in think_parts)
        return ParsedResponse(
            content=content or "",
            reasoning_content=reasoning,
            tool_calls=normalize_tool_calls(message.get("tool_calls")),
        )

    def _close_token_ids(self) -> list[int]:
        """Turn-close (stop) token ids, e.g. ``<|im_end|>`` / EOS. Cached."""
        if self._close_ids is not None:
            return self._close_ids
        ids: list[int] = []
        unk = getattr(self._tokenizer, "unk_token_id", None)
        for stop in self._renderer.get_stop_sequences():
            if isinstance(stop, int):
                ids.append(stop)
                continue
            # String stop -> token id. Prefer a single special-token id;
            # otherwise encode and keep only if it is exactly one token.
            tid = self._tokenizer.convert_tokens_to_ids(stop)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.append(tid)
                continue
            encoded = self._tokenizer.encode(stop, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
            else:
                logger.debug("Dropping multi-token stop sequence %r (%d tokens)", stop, len(encoded))
        self._close_ids = ids
        return ids

    def get_stop_token_ids(self) -> list[int]:
        return list(self._close_token_ids())

    def get_stop_sequences(self) -> list[str] | list[int]:
        """Passthrough for callers that want raw stop strings/ids (not in the
        native Protocol, but several rLLM engines consume string stops)."""
        return self._renderer.get_stop_sequences()

    @staticmethod
    def _trim_to_turn_close(
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        close_ids: set[int],
        synthesize_close: int | None,
    ) -> list[int] | None:
        """Longest prefix of ``prev_prompt + prev_completion`` ending at a
        turn-close token (scanning only within the completion), or — on a
        truncated prior turn with no close — the sequence plus a synthesized
        close. ``None`` if neither is possible. Mirrors prime-rl's
        ``trim_to_turn_close``."""
        ids = list(previous_prompt_ids) + list(previous_completion_ids)
        for idx in range(len(ids) - 1, len(previous_prompt_ids) - 1, -1):
            if ids[idx] in close_ids:
                return ids[: idx + 1]
        if synthesize_close is None:
            return None
        return ids + [synthesize_close]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
    ) -> RenderedTokens | None:
        """Extend the prior turn's verbatim tokens with the new turn's framing.

        Returns ``None`` (caller falls back to a full re-render) when the
        prefix-extension contract can't be proven: assistant content in the new
        slice (would re-tokenize sampled tokens), a truncated prior with no
        recoverable close token, or multimodal content.
        """
        import tinker
        from tinker_cookbook.renderers.base import RenderContext

        if not previous_prompt_ids or not new_messages:
            return None
        if any(m.get("role") == "assistant" for m in new_messages):
            return None
        if any(not isinstance(m.get("content", ""), str) for m in new_messages):
            return None
        close_ids = self._close_token_ids()
        if not close_ids:
            return None

        anchor = self._trim_to_turn_close(
            previous_prompt_ids, previous_completion_ids, set(close_ids), close_ids[0]
        )
        if anchor is None:
            return None

        # Delta: render only the new (non-assistant) messages + the next
        # assistant opener, using the same primitives build_generation_prompt
        # uses. The new slice always follows the just-sampled assistant turn, so
        # the first new message's render context sees a prior assistant (matters
        # for renderers that group tool turns or vary inter-turn separators).
        n_before = 2
        full_len = n_before + len(new_messages)
        last_user_index = max(
            (n_before + j for j, m in enumerate(new_messages) if m.get("role") == "user"),
            default=n_before - 1,
        )
        prior_assistant = {"role": "assistant", "content": ""}

        delta_chunks: list[Any] = []
        for j, msg in enumerate(new_messages):
            ctx = RenderContext(
                idx=n_before + j,
                is_last=False,
                prev_message=new_messages[j - 1] if j > 0 else prior_assistant,
                last_user_index=last_user_index,
            )
            rm = self._renderer.render_message(msg, ctx)
            if rm.header:
                delta_chunks.append(rm.header)
            delta_chunks.extend(
                c
                for c in rm.output
                if not (isinstance(c, tinker.EncodedTextChunk) and not c.tokens)
            )

        suffix_ctx = RenderContext(
            idx=full_len,
            is_last=True,
            prev_message=new_messages[-1],
            last_user_index=last_user_index,
        )
        suffix_tokens = self._renderer._get_generation_suffix("assistant", suffix_ctx)

        delta = tinker.ModelInput(chunks=delta_chunks).to_ints() + list(suffix_tokens)
        return RenderedTokens(token_ids=anchor + delta)


def make_tinker_renderer(
    name: str,
    tokenizer: Any,
    image_processor: Any | None = None,
) -> TinkerAdapter:
    """Build a :class:`TinkerAdapter` via tinker-cookbook ``get_renderer``.

    Imports the Fireworks cookbook registrations as a side effect so names like
    ``deepseek_v4`` / ``gemma4`` resolve.
    """
    if not TINKER_AVAILABLE:
        raise RuntimeError(
            "tinker_cookbook is not installed. Install it with: pip install tinker-cookbook"
        )
    from tinker_cookbook import renderers as tc_renderers

    from rllm.renderers._fw_register import ensure_registered

    ensure_registered()  # make cookbook renderers (deepseek_v4, gemma4, …) resolvable
    renderer = tc_renderers.get_renderer(name, tokenizer, image_processor=image_processor)
    return TinkerAdapter(renderer, tokenizer, renderer_name=name)
