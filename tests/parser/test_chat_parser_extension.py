"""Property (iv) — sequence-extension — for rLLM chat-template parsers.

For every realistic engine-input transition ``T_k -> T_{k+1}`` in the canonical
multi-turn fixture, this test asserts:

    tokenize(parse(T_{k+1}, add_generation_prompt=True))
        startswith tokenize(parse(T_k, add_generation_prompt=True))

"Realistic" means both endpoints end on user or tool (never on assistant) —
the engine is never asked to produce a generation prompt for a state that
ends on its own last reply.

These tests are deliberately tokenizer-real (Qwen3-0.6B locally cached) and
skip cleanly if the tokenizer/model is unavailable. The fixture lives in
``tests/parser/_fixtures/multi_turn_chain.py``.
"""

from __future__ import annotations

import pytest
from _fixtures.multi_turn_chain import (
    get_chain,
    get_realistic_transitions,
    step_by_label,
)


def _load_qwen_parser():
    """Lazy-load Qwen3-0.6B tokenizer + QwenChatTemplateParser.

    Returns None (test skipped) if transformers isn't importable or the
    model isn't cached locally and the environment has no network.
    """
    try:
        from transformers import AutoTokenizer  # noqa: WPS433 (lazy)
    except ImportError:
        pytest.skip("transformers not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — broad, but want any failure to skip
        pytest.skip(f"Qwen/Qwen3-0.6B tokenizer not available: {exc}")

    from rllm.parser import QwenChatTemplateParser

    return QwenChatTemplateParser(tokenizer)


def _tokenize_step(parser, step) -> list[int]:
    rendered = parser.parse(
        step["messages"],
        add_generation_prompt=True,
        accumulate_reasoning=True,
        is_first_msg=True,
    )
    return parser.tokenizer.encode(rendered, add_special_tokens=False)


@pytest.mark.parametrize("variant", ["ending_on_tool", "ending_on_user"])
def test_qwen_parser_extension_property_on_realistic_transitions(variant):
    parser = _load_qwen_parser()
    chain = get_chain(variant)
    transitions = get_realistic_transitions(variant)
    assert transitions, f"no realistic transitions defined for variant={variant}"

    for src_label, dst_label in transitions:
        src = step_by_label(chain, src_label)
        dst = step_by_label(chain, dst_label)
        src_tokens = _tokenize_step(parser, src)
        dst_tokens = _tokenize_step(parser, dst)

        assert len(dst_tokens) > len(src_tokens), f"{variant}: {src_label} -> {dst_label}: expected dst tokens to be longer than src (src={len(src_tokens)}, dst={len(dst_tokens)})"
        prefix = dst_tokens[: len(src_tokens)]
        assert prefix == src_tokens, (
            f"{variant}: {src_label} -> {dst_label}: "
            f"token-level extension property failed. "
            f"first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(prefix, src_tokens, strict=True)) if a != b), '?')}"
        )


def test_qwen_parser_each_step_renders_without_error():
    """Smoke: every step (including the assistant-ending intermediate ones)
    must be parseable — even though we don't require the extension property
    on assistant-ending prefixes, we DO require they don't crash."""
    parser = _load_qwen_parser()
    for variant in ("ending_on_tool", "ending_on_user"):
        chain = get_chain(variant)
        for step in chain:
            rendered = parser.parse(
                step["messages"],
                add_generation_prompt=True,
                accumulate_reasoning=True,
                is_first_msg=True,
            )
            assert isinstance(rendered, str) and len(rendered) > 0, f"{variant}:{step['label']} rendered empty"
