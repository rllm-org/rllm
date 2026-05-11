"""Property (i)′ — Tinker oracle equality for the Qwen chat-template parser.

For the canonical multi-turn fixture, compare token-level output between
``rllm.parser.QwenChatTemplateParser`` and Tinker's
``Qwen3Renderer(strip_thinking_from_history=False)``. The criterion (see
``tmp/chat_parser_refactor_loop/remarks.md``):

* **Strict equality** on messages that do NOT contain ``reasoning``. rLLM
  and Tinker should produce byte/token-identical output here, modulo a BOS
  difference if either renderer prepends one.
* **Documented divergence** on messages WITH ``reasoning``:
    * rLLM emits ``<think>\\n{r}\\n</think>\\n\\n`` (newline-wrapped).
    * Tinker emits ``<think>{r}</think>`` (inline).
  The token diff in this case is bounded — the test verifies that the
  divergence is *only* in the think-block region by reconciling rLLM's
  string to Tinker's and confirming the rewritten form matches byte-for-byte.

This test skips if either Tinker or the Qwen3-0.6B tokenizer is not
available locally — it's intended to be informational, not gating.
"""

from __future__ import annotations

import json

import pytest
from _fixtures.multi_turn_chain import (
    CHAIN_ENDING_ON_TOOL,
    get_chain,
)

# ── Loader ────────────────────────────────────────────────────────────────


def _load_parsers():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3-0.6B tokenizer not available: {exc}")

    try:
        from tinker_cookbook.renderers.qwen3 import Qwen3Renderer  # noqa: WPS433
    except ImportError as exc:
        pytest.skip(f"tinker-cookbook not available: {exc}")

    from rllm.parser import QwenChatTemplateParser

    rllm_parser = QwenChatTemplateParser(tokenizer)
    tinker_renderer = Qwen3Renderer(tokenizer, strip_thinking_from_history=False)
    return rllm_parser, tinker_renderer, tokenizer


# ── Conversion: rLLM messages → Tinker messages ───────────────────────────


def _rllm_to_tinker_messages(messages: list[dict]) -> list[dict]:
    """Convert rLLM-shaped messages to Tinker-shaped messages.

    Key differences:
    - ``reasoning`` (str on assistant) → ``ThinkingPart`` inside the
      ``content`` list (Tinker treats reasoning as a content part).
    - ``tool_calls`` (OpenAI-shape dicts) → pydantic ``ToolCall`` instances.
    - Tool messages: pass through ``content``, ``tool_call_id``, ``name``.
    """
    from tinker_cookbook.renderers.base import TextPart, ThinkingPart, ToolCall

    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "assistant":
            content_str = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            tool_calls = msg.get("tool_calls") or []

            if reasoning:
                # Tinker wants reasoning as a ThinkingPart inside content.
                parts: list = []
                parts.append(ThinkingPart(type="thinking", thinking=reasoning))
                if content_str:
                    parts.append(TextPart(type="text", text=content_str))
                new_content: object = parts
            else:
                new_content = content_str

            new_msg: dict = {"role": "assistant", "content": new_content}
            if tool_calls:
                tinker_tcs: list[ToolCall] = []
                for tc in tool_calls:
                    fn = tc.get("function") or tc
                    name = fn["name"]
                    args = fn["arguments"]
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    tinker_tcs.append(
                        ToolCall(
                            id=tc.get("id"),
                            function=ToolCall.FunctionBody(name=name, arguments=args),
                        )
                    )
                new_msg["tool_calls"] = tinker_tcs
            out.append(new_msg)
        elif role == "tool":
            new_msg = {"role": "tool", "content": msg.get("content", "")}
            if "tool_call_id" in msg:
                new_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                new_msg["name"] = msg["name"]
            out.append(new_msg)
        else:
            out.append({"role": role, "content": msg.get("content", "")})
    return out


def _tinker_render_tokens(tinker_renderer, messages: list[dict]) -> list[int]:
    """Render via Tinker and flatten the resulting chunked token output."""
    import tinker

    model_input = tinker_renderer.build_generation_prompt(messages, role="assistant")
    flat: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            flat.extend(chunk.tokens)
        else:
            raise AssertionError(f"unexpected chunk type {type(chunk).__name__}")
    return flat


def _rllm_render_tokens(rllm_parser, messages: list[dict]) -> list[int]:
    rendered = rllm_parser.parse(
        messages,
        add_generation_prompt=True,
        is_first_msg=True,
        accumulate_reasoning=True,
    )
    return rllm_parser.tokenizer.encode(rendered, add_special_tokens=False)


def _any_message_has_reasoning(messages: list[dict]) -> bool:
    return any((m.get("role") == "assistant" and (m.get("reasoning") or "")) for m in messages)


# ── Test cases ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("variant", "step_label"),
    [
        ("ending_on_tool", "S0_system_plus_user"),
        ("ending_on_user", "S0_system_plus_user"),
    ],
)
def test_tinker_oracle_strict_equality_no_reasoning(variant, step_label):
    """For the system+user-only stage (no reasoning anywhere), rLLM tokens
    should exactly equal Tinker tokens."""
    rllm_parser, tinker_renderer, tokenizer = _load_parsers()
    chain = get_chain(variant)
    step = next(s for s in chain if s["label"] == step_label)
    assert not _any_message_has_reasoning(step["messages"]), "this test case is only meaningful when no reasoning is present"

    rllm_ids = _rllm_render_tokens(rllm_parser, step["messages"])
    tinker_ids = _tinker_render_tokens(tinker_renderer, _rllm_to_tinker_messages(step["messages"]))

    if rllm_ids != tinker_ids:
        # Surface a useful diff on failure.
        rllm_str = tokenizer.decode(rllm_ids, skip_special_tokens=False)
        tinker_str = tokenizer.decode(tinker_ids, skip_special_tokens=False)
        # Find first divergence
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(rllm_str, tinker_str, strict=False)) if a != b),
            min(len(rllm_str), len(tinker_str)),
        )
        raise AssertionError(
            f"rLLM and Tinker token streams diverged at char {first_diff}:\n"
            f"  rllm[{first_diff}:{first_diff + 40}] = {rllm_str[first_diff : first_diff + 40]!r}\n"
            f"  tnkr[{first_diff}:{first_diff + 40}] = {tinker_str[first_diff : first_diff + 40]!r}"
        )


@pytest.mark.parametrize(
    ("variant", "step_label"),
    [
        ("ending_on_tool", "S2_after_tool_response"),
        ("ending_on_user", "S2_after_tool_response"),
        ("ending_on_user", "S4_after_followup_user"),
    ],
)
def test_tinker_oracle_documented_divergence_with_reasoning(variant, step_label):
    """When ``reasoning`` is present, rLLM and Tinker diverge ONLY in the
    think-block formatting. The divergence is bounded: replacing rLLM's
    ``<think>\\n{r}\\n</think>\\n\\n`` with Tinker's ``<think>{r}</think>``
    should make the two strings byte-equal."""
    rllm_parser, tinker_renderer, _ = _load_parsers()
    chain = get_chain(variant)
    step = next(s for s in chain if s["label"] == step_label)
    assert _any_message_has_reasoning(step["messages"]), "this test case is only meaningful when reasoning is present"

    rllm_str = rllm_parser.parse(
        step["messages"],
        add_generation_prompt=True,
        is_first_msg=True,
        accumulate_reasoning=True,
    )

    # Rewrite rLLM's newline-wrapped think blocks to Tinker's inline form.
    import re as _re

    rewritten = _re.sub(
        r"<think>\n(.*?)\n</think>\n\n",
        lambda m: f"<think>{m.group(1)}</think>",
        rllm_str,
        flags=_re.DOTALL,
    )

    tinker_ids = _tinker_render_tokens(tinker_renderer, _rllm_to_tinker_messages(step["messages"]))
    tinker_str = rllm_parser.tokenizer.decode(tinker_ids, skip_special_tokens=False)

    assert rewritten == tinker_str, f"After normalizing <think> formatting, rLLM and Tinker still diverge:\n  rewritten[-200:] = {rewritten[-200:]!r}\n  tinker_str[-200:] = {tinker_str[-200:]!r}"


def test_tinker_oracle_token_count_within_bounds():
    """Sanity: at S2 of the ending_on_tool variant, the token-count gap
    between rLLM and Tinker should be small (just the per-think-block
    newline overhead, ~4 tokens per assistant with reasoning)."""
    rllm_parser, tinker_renderer, _ = _load_parsers()
    step = next(s for s in CHAIN_ENDING_ON_TOOL if s["label"] == "S2_after_tool_response")

    rllm_ids = _rllm_render_tokens(rllm_parser, step["messages"])
    tinker_ids = _tinker_render_tokens(tinker_renderer, _rllm_to_tinker_messages(step["messages"]))

    diff = len(rllm_ids) - len(tinker_ids)
    # Each newline-wrapped <think> contributes 4 extra characters
    # ('\n' before {r}, '\n' before </think>, two '\n' after </think>).
    # Tokenized, that's roughly 4 tokens per assistant with reasoning.
    # S2 has one such assistant, so cap at 6 tokens to give some slack.
    assert 0 <= diff <= 6, f"unexpected token-count delta between rLLM and Tinker: rllm={len(rllm_ids)}, tinker={len(tinker_ids)}, diff={diff}"
