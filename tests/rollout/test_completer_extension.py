"""Property (ii)′ — TITOCompleter state invariant — across a multi-turn rollout.

After ``TITOCompleter.complete()`` finishes turn k, the invariant is:

    decode(_prev_token_input) == _prev_messages_str
    _prev_messages_str.startswith(parse(messages, add_generation_prompt=True))

i.e. ``_prev_messages_str`` is the string form of the same byte sequence that
``_prev_token_input`` represents, and it extends the conversation history
through the just-generated assistant turn.

The Part 1 report
(``tmp/parser_prefix_tests/REPORT.md`` §"Pathway 3") showed the pre-fix
completer would silently double-count the assistant turn at turn k+1 by ~30
tokens. This test pins the post-fix invariant and the multi-turn extension
property that follows from it.

The test uses a stub ``RolloutEngine`` whose ``get_token_output_from_token_input``
returns a pre-canned set of completion ids — so we exercise the completer's
state machine without standing up a real model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Sibling-import path for the canonical fixture lives under tests/parser/_fixtures.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parser"))
from _fixtures.multi_turn_chain import (  # noqa: E402
    ASSISTANT_1,
    ASSISTANT_2_FINAL,
    TOOL_RESPONSE_1,
    USER_QUESTION,
)

# ── Stub RolloutEngine ─────────────────────────────────────────────────────


@dataclass
class _StubTokenOutput:
    """Mimics verl-style token output (has .token_ids)."""

    token_ids: list[int]
    logprobs: list[float]


class _StubRolloutEngine:
    """Minimal RolloutEngine for completer tests.

    Returns pre-canned completion ids on each call. The list of canned
    completions is pushed by the test; each ``get_token_output_from_token_input``
    consumes the next one.
    """

    is_validation = False

    def __init__(self, tokenizer, chat_parser, canned_completions: list[list[int]]):
        self.tokenizer = tokenizer
        self.chat_parser = chat_parser
        self._canned = list(canned_completions)
        self.weight_version = 0
        self.last_token_inputs: list[list[int]] = []

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    async def get_token_output_from_token_input(self, token_input, **kwargs):
        self.last_token_inputs.append(list(token_input))
        if not self._canned:
            raise RuntimeError("stub rollout: ran out of canned completions")
        completion_ids = self._canned.pop(0)
        return _StubTokenOutput(token_ids=completion_ids, logprobs=[0.0] * len(completion_ids))

    def assemble_model_output(self, token_input, token_output):
        from rllm.experimental.rollout.rollout_engine import ModelOutput

        text = self.tokenizer.decode(token_output.token_ids, skip_special_tokens=False)
        return ModelOutput(
            text=text,
            content=text,
            reasoning=None,
            tool_calls=None,
            prompt_ids=token_input,
            completion_ids=token_output.token_ids,
            prompt_length=len(token_input),
            completion_length=len(token_output.token_ids),
        )


def _load_parser_and_tokenizer():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3-0.6B not available: {exc}")
    from rllm.parser import QwenChatTemplateParser

    return QwenChatTemplateParser(tokenizer), tokenizer


def _make_canned_completion_from_assistant(parser, tokenizer, asst_msg: dict) -> list[int]:
    """Render an assistant message body via the parser, then tokenize.

    The returned ids represent what a "verbatim" model would emit if it
    produced exactly the given assistant message — i.e. everything between
    the assistant header and the eot newline. ``parse_assistant`` returns
    ``assistant_token + body + eot_token`` where eot is ``<|im_end|>\\n``;
    the model would stop AT ``<|im_end|>``, so we strip the trailing ``\\n``.
    """
    full = parser.parse_assistant(asst_msg, accumulate_reasoning=True)
    assert full.startswith(parser.assistant_token)
    body_with_eot = full[len(parser.assistant_token) :]
    # eot_token is "<|im_end|>\n"; the model stops at "<|im_end|>" (no trailing newline).
    assert body_with_eot.endswith(parser.eot_token), f"unexpected suffix: {body_with_eot[-20:]!r}"
    body_to_im_end = body_with_eot[: -len("\n")]  # strip just the trailing newline
    return tokenizer.encode(body_to_im_end, add_special_tokens=False)


def test_completer_state_invariant_after_each_turn():
    import asyncio

    asyncio.run(_async_test_completer_state_invariant_after_each_turn())


async def _async_test_completer_state_invariant_after_each_turn():
    """After every complete() call:

    1. decode(_prev_token_input) == _prev_messages_str   (state sync)
    2. _prev_messages_str ends with the canned completion's decoded form
    3. Next turn's input still starts with the previous turn's _prev_messages_str
       (the byte-level extension property the completer exists to provide)
    """
    parser, tokenizer = _load_parser_and_tokenizer()

    canned_1 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_1)
    canned_2 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_2_FINAL)

    engine = _StubRolloutEngine(
        tokenizer=tokenizer,
        chat_parser=parser,
        canned_completions=[canned_1, canned_2],
    )

    from rllm.experimental.rollout.completer import TITOCompleter

    completer = TITOCompleter(engine)

    # ── Turn 1: messages = [sys, user] ────────────────────────────────────
    messages_t1 = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": USER_QUESTION},
    ]
    await completer.complete(messages_t1)

    decoded_state_1 = tokenizer.decode(completer._prev_token_input, skip_special_tokens=False)
    assert decoded_state_1 == completer._prev_messages_str, (
        f"After turn 1, decode(_prev_token_input) != _prev_messages_str.\n  decoded ends with: ...{decoded_state_1[-100:]!r}\n  state ends with:   ...{completer._prev_messages_str[-100:]!r}"
    )

    # ── Turn 2: messages = [sys, user, asst_1, tool_response_1] ───────────
    messages_t2 = [
        *messages_t1,
        ASSISTANT_1,
        TOOL_RESPONSE_1,
    ]
    prev_messages_str_after_turn1 = completer._prev_messages_str
    prev_token_input_after_turn1 = list(completer._prev_token_input)

    await completer.complete(messages_t2)

    # State invariant after turn 2.
    decoded_state_2 = tokenizer.decode(completer._prev_token_input, skip_special_tokens=False)
    assert decoded_state_2 == completer._prev_messages_str, (
        f"After turn 2, decode(_prev_token_input) != _prev_messages_str.\n  decoded ends with: ...{decoded_state_2[-100:]!r}\n  state ends with:   ...{completer._prev_messages_str[-100:]!r}"
    )

    # Byte-level extension: turn 2's state must extend turn 1's state.
    assert completer._prev_messages_str.startswith(prev_messages_str_after_turn1), "Turn 2 _prev_messages_str does not extend turn 1's _prev_messages_str."
    # Token-level extension: turn 2's token buffer must extend turn 1's.
    assert completer._prev_token_input[: len(prev_token_input_after_turn1)] == prev_token_input_after_turn1, "Turn 2 _prev_token_input does not extend turn 1's _prev_token_input."


def test_completer_prefix_check_succeeds_after_state_fix():
    import asyncio

    asyncio.run(_async_test_completer_prefix_check_succeeds_after_state_fix())


async def _async_test_completer_prefix_check_succeeds_after_state_fix():
    """The whole purpose of the completer's state machine: after fixing the
    state sync, a multi-turn rollout should report ``n_prefixes >= 1``.

    Pre-fix, _prev_messages_str ended in the gen-prompt while _prev_token_input
    extended past the asst body; the prefix check on turn 2 would still
    accept because the gen-prompt is a substring of cur_messages_str, but the
    resulting curr_token_input would double-count the asst body. Post-fix,
    both branches stay consistent and ``n_prefixes`` increments correctly.
    """
    parser, tokenizer = _load_parser_and_tokenizer()

    canned_1 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_1)
    canned_2 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_2_FINAL)
    engine = _StubRolloutEngine(
        tokenizer=tokenizer,
        chat_parser=parser,
        canned_completions=[canned_1, canned_2],
    )

    from rllm.experimental.rollout.completer import TITOCompleter

    completer = TITOCompleter(engine)
    messages_t1 = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": USER_QUESTION},
    ]
    await completer.complete(messages_t1)
    await completer.complete([*messages_t1, ASSISTANT_1, TOOL_RESPONSE_1])

    assert completer.n_completions == 2
    assert completer.n_prefixes >= 1, f"expected at least 1 prefix hit across 2 turns, got n_prefixes={completer.n_prefixes}"


def test_completer_no_token_double_count():
    import asyncio

    asyncio.run(_async_test_completer_no_token_double_count())


def test_completer_forwards_tools_to_parser():
    """``tools=`` passed to ``complete()`` must reach the chat parser so the
    tool prompt is injected into the system message. Pre-fix, it was
    forwarded to the engine instead, silently bypassing the parser.

    Witness: the rendered ``_prev_messages_str`` after one turn must contain
    the canonical ``# Tools`` header that the QwenToolParser injects.
    """
    import asyncio

    asyncio.run(_async_test_completer_forwards_tools_to_parser())


async def _async_test_completer_forwards_tools_to_parser():
    parser, tokenizer = _load_parser_and_tokenizer()

    canned = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_1)
    engine = _StubRolloutEngine(
        tokenizer=tokenizer,
        chat_parser=parser,
        canned_completions=[canned],
    )

    from rllm.experimental.rollout.completer import TITOCompleter

    completer = TITOCompleter(engine)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "You are a math helper."},
        {"role": "user", "content": USER_QUESTION},
    ]
    await completer.complete(messages, tools=tools)

    state = completer._prev_messages_str
    assert "# Tools" in state, (
        "completer did not forward tools= to chat parser — the system message "
        "is missing the '# Tools' header. The model would never see the tool "
        "definitions and training would converge to 'never call tools'."
    )
    assert '"calculate"' in state or "calculate" in state, "tool definition for 'calculate' is missing from the rendered prompt."


async def _async_test_completer_no_token_double_count():
    """At turn 2, the engine's input token count must equal a fresh static
    encode of the full message chain WITH gen prompt — no double-counting.
    """
    parser, tokenizer = _load_parser_and_tokenizer()
    canned_1 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_1)
    canned_2 = _make_canned_completion_from_assistant(parser, tokenizer, ASSISTANT_2_FINAL)
    engine = _StubRolloutEngine(
        tokenizer=tokenizer,
        chat_parser=parser,
        canned_completions=[canned_1, canned_2],
    )

    from rllm.experimental.rollout.completer import TITOCompleter

    completer = TITOCompleter(engine)
    messages_t1 = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": USER_QUESTION},
    ]
    await completer.complete(messages_t1)
    messages_t2 = [*messages_t1, ASSISTANT_1, TOOL_RESPONSE_1]
    await completer.complete(messages_t2)

    # Engine's most recent input is the curr_token_input for turn 2.
    actual_input_t2 = engine.last_token_inputs[-1]

    # Compare against a fresh static encode: parse(messages_t2_with_asst) + gen prompt.
    # That is the bytes the engine SHOULD see if there were no double-count.
    fresh_str = parser.parse(messages_t2, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=True)
    fresh_ids = tokenizer.encode(fresh_str, add_special_tokens=False)

    assert len(actual_input_t2) == len(fresh_ids), (
        f"completer token-input length disagrees with a fresh static encode: "
        f"completer={len(actual_input_t2)}, fresh={len(fresh_ids)} "
        f"(pre-fix would show ~+30 tokens drift; if you see that, the completer regressed)"
    )
    assert actual_input_t2 == fresh_ids, (
        "completer-built token input is not byte-equal to a fresh static encode; "
        "first divergence at idx "
        f"{next((i for i, (a, b) in enumerate(zip(actual_input_t2, fresh_ids, strict=True)) if a != b), '?')}"
    )
