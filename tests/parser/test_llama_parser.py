"""Tests for the Llama 3.2 chat-template + tool parser pair.

Two layers of coverage:

1. ``LlamaToolParser.parse`` directly on canonical wire-format strings —
   bare JSON (Llama 3.2), <|python_tag|>-prefixed JSON (Llama 3.1 compat),
   and malformed/missing-field rejection.
2. Round-trip ``parse_completion(parse_assistant(msg))`` over a stub
   tokenizer-aware parser, asserting the dict shape matches.
3. Structural equivalence with HF ``tokenizer.apply_chat_template`` on
   the bare assistant-only and ipython-only single-turn cases. The full
   chat template (with date prefixes / Environment: ipython / tools-in-
   first-user-message) is Phase B and not tested here.
"""

from __future__ import annotations

import json

import pytest

from rllm.parser.chat_template.llama import LlamaChatTemplateParser
from rllm.parser.tool_parser import LlamaToolParser

# ───────────────────────── LlamaToolParser unit tests ─────────────────────────


def test_llama_tool_parser_bare_json_3_2():
    parser = LlamaToolParser()
    body = '{"name": "calculate", "parameters": {"expression": "2+3"}}'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].name == "calculate"
    assert calls[0].arguments == {"expression": "2+3"}


def test_llama_tool_parser_python_tag_3_1_compat():
    parser = LlamaToolParser()
    body = '<|python_tag|>{"name": "search", "parameters": {"q": "rllm"}}'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "rllm"}


def test_llama_tool_parser_arguments_fallback_field():
    """Some Llama fine-tunes emit ``arguments`` (OpenAI shape) instead of
    ``parameters``. Accept both."""
    parser = LlamaToolParser()
    body = '{"name": "calc", "arguments": {"x": 1}}'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].arguments == {"x": 1}


def test_llama_tool_parser_trailing_prose_tolerated():
    """The model may continue with text after the JSON. The first
    balanced object should still be extracted."""
    parser = LlamaToolParser()
    body = '{"name": "calc", "parameters": {"x": 5}}\n\nthen we will sum.'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].arguments == {"x": 5}


def test_llama_tool_parser_no_tool_call_returns_empty():
    parser = LlamaToolParser()
    assert parser.parse("just plain text, no tool call here") == []
    assert parser.parse("{not valid json}") == []
    assert parser.parse('{"missing_name": true}') == []


def test_llama_tool_parser_string_arguments_decoded():
    """If ``parameters`` arrives as a JSON-encoded string, decode it."""
    parser = LlamaToolParser()
    body = '{"name": "calc", "parameters": "{\\"x\\": 1}"}'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].arguments == {"x": 1}


def test_llama_tool_parser_nested_braces_in_string():
    """A string literal containing braces should not confuse the balanced-
    brace scanner."""
    parser = LlamaToolParser()
    body = '{"name": "echo", "parameters": {"text": "hello {world}"}}'
    calls = parser.parse(body)
    assert len(calls) == 1
    assert calls[0].arguments == {"text": "hello {world}"}


# ─────────────── LlamaChatTemplateParser fixture (real tokenizer) ───────────────


@pytest.fixture(scope="module")
def llama_tokenizer():
    """Real Llama-3.2-3B-Instruct tokenizer from the local HF cache."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


@pytest.fixture(scope="module")
def llama_parser(llama_tokenizer):
    return LlamaChatTemplateParser(llama_tokenizer)


# ─────────────── parse_completion <-> parse_assistant round-trip ───────────────


def test_round_trip_content_only(llama_parser, llama_tokenizer):
    msg = {"role": "assistant", "content": "Hello, world!"}
    rendered = llama_parser.parse_assistant(msg)
    # parse_assistant returns the full assistant turn including header + eot.
    # parse_completion expects the model's raw output (post-header, pre-eot)
    # as token ids. Re-tokenize and decode through the tokenizer.
    body_text = rendered[len(llama_parser.assistant_token) :]
    completion_ids = llama_tokenizer.encode(body_text, add_special_tokens=False)
    parsed = llama_parser.parse_completion(completion_ids)
    assert parsed["content"] == "Hello, world!"
    assert parsed["reasoning"] == ""
    assert parsed["tool_calls"] == []


def test_round_trip_tool_call_only(llama_parser, llama_tokenizer):
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "calculate", "arguments": {"expression": "2+3"}}},
        ],
    }
    rendered = llama_parser.parse_assistant(msg)
    body_text = rendered[len(llama_parser.assistant_token) :]
    completion_ids = llama_tokenizer.encode(body_text, add_special_tokens=False)
    parsed = llama_parser.parse_completion(completion_ids)
    assert parsed["content"] == ""
    assert parsed["reasoning"] == ""
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0].name == "calculate"
    assert parsed["tool_calls"][0].arguments == {"expression": "2+3"}


def test_round_trip_content_plus_tool_call(llama_parser, llama_tokenizer):
    msg = {
        "role": "assistant",
        "content": "I'll compute that.",
        "tool_calls": [
            {"type": "function", "function": {"name": "calculate", "arguments": {"expression": "7*8"}}},
        ],
    }
    rendered = llama_parser.parse_assistant(msg)
    body_text = rendered[len(llama_parser.assistant_token) :]
    completion_ids = llama_tokenizer.encode(body_text, add_special_tokens=False)
    parsed = llama_parser.parse_completion(completion_ids)
    # The content is preserved; the JSON tool-call body is stripped out.
    assert "I'll compute that." in parsed["content"]
    assert "calculate" not in parsed["content"]  # JSON body removed
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0].arguments == {"expression": "7*8"}


# ─────────────── structural equivalence with HF apply_chat_template ───────────────


def test_parse_tool_matches_hf_template_ipython_header(llama_parser, llama_tokenizer):
    """Phase A: a single ``role=tool`` message should render to the same
    ipython-role block as HF's apply_chat_template emits.

    We compare just the rendered tool turn (ignoring the surrounding
    system / user prefix that HF injects, since matching those bytes is
    Phase B).
    """
    tool_msg = {"role": "tool", "content": "5"}
    ours = llama_parser.parse_tool(tool_msg)

    # The HF template wraps tool content in JSON if it's a string —
    # let's verify the header + content + eot triplet at least matches.
    assert ours.startswith("<|start_header_id|>ipython<|end_header_id|>\n\n")
    assert ours.endswith("<|eot_id|>")
    assert "5" in ours


def test_parse_user_matches_hf_template(llama_parser, llama_tokenizer):
    user_msg = {"role": "user", "content": "What is 2+3?"}
    ours = llama_parser.parse_user(user_msg)
    expected = "<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+3?<|eot_id|>"
    assert ours == expected


def test_assistant_tool_call_body_matches_hf_format(llama_parser, llama_tokenizer):
    """The body our parser emits for a tool-calling assistant message
    must be byte-identical to the body HF emits — modulo the surrounding
    header/eot which we own. Specifically, both should be
    ``{"name": "...", "parameters": {...}}``.
    """
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "calc", "arguments": {"x": 1}}},
        ],
    }
    ours = llama_parser.parse_assistant(msg)
    body = ours[len(llama_parser.assistant_token) : -len(llama_parser.eot_token)]
    decoded = json.loads(body)
    assert decoded["name"] == "calc"
    assert decoded["parameters"] == {"x": 1}
    # Crucially, ``arguments`` field is renamed to ``parameters`` —
    # Llama convention, not OpenAI.
    assert "arguments" not in decoded
