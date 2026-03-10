import sys
from unittest.mock import patch

import pytest
from tinker_cookbook import renderers
from transformers import AutoTokenizer

from rllm.parser import QwenChatTemplateParser
from rllm.parser.tinker_parser import TinkerChatTemplateParser
from rllm.parser.utils import SIMPLE_TEST_MESSAGES


@pytest.fixture
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


@pytest.fixture
def qwen_renderer(qwen_tokenizer):
    return renderers.get_renderer("qwen3", qwen_tokenizer)


@pytest.fixture
def qwen_tinker_parser(qwen_renderer):
    return TinkerChatTemplateParser(qwen_renderer)


def test_tinker_parser_init(qwen_tinker_parser):
    """Verify that constructor sets up generation_prompt and stop_sequences."""
    assert qwen_tinker_parser.generation_prompt
    assert isinstance(qwen_tinker_parser.generation_prompt, str)
    assert qwen_tinker_parser.stop_sequences is not None
    assert qwen_tinker_parser.tokenizer is not None
    assert qwen_tinker_parser.renderer is not None


def test_tinker_parser_init_bad_renderer():
    """Verify TypeError when passing a non-renderer object."""
    with pytest.raises(TypeError, match="Expected a tinker_cookbook Renderer"):
        TinkerChatTemplateParser("not a renderer")


def test_tinker_parser_parse(qwen_tinker_parser):
    """Verify parse() returns a valid non-empty string."""
    result = qwen_tinker_parser.parse(SIMPLE_TEST_MESSAGES, add_generation_prompt=True, is_first_msg=True)
    assert isinstance(result, str)
    assert len(result) > 0


def test_tinker_parser_parse_empty():
    """Verify parse([]) returns empty string."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    renderer = renderers.get_renderer("qwen3", tokenizer)
    parser = TinkerChatTemplateParser(renderer)
    assert parser.parse([]) == ""


def test_tinker_parser_parse_generation_prompt(qwen_tinker_parser):
    """Verify that generation prompt is appended when requested."""
    with_prompt = qwen_tinker_parser.parse(SIMPLE_TEST_MESSAGES, add_generation_prompt=True, is_first_msg=True)
    without_prompt = qwen_tinker_parser.parse(SIMPLE_TEST_MESSAGES, add_generation_prompt=False, is_first_msg=True)
    # The version with generation prompt should be longer
    assert len(with_prompt) > len(without_prompt)


def test_tinker_parser_parse_is_first_msg(qwen_tinker_parser):
    """Verify is_first_msg controls BOS token inclusion."""
    with_bos = qwen_tinker_parser.parse(SIMPLE_TEST_MESSAGES, is_first_msg=True)
    without_bos = qwen_tinker_parser.parse(SIMPLE_TEST_MESSAGES, is_first_msg=False)
    # With BOS should be at least as long as without
    assert len(with_bos) >= len(without_bos)


def test_tinker_parser_parse_with_reasoning(qwen_tinker_parser):
    """Verify that reasoning is included when accumulate_reasoning=True."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there", "reasoning": "The user greeted me"},
    ]
    with_reasoning = qwen_tinker_parser.parse(messages, accumulate_reasoning=True, is_first_msg=True)
    without_reasoning = qwen_tinker_parser.parse(messages, accumulate_reasoning=False, is_first_msg=True)
    assert "think" in with_reasoning or len(with_reasoning) > len(without_reasoning)


def test_tinker_parser_parse_completion(qwen_tinker_parser, qwen_tokenizer):
    """Verify parse_completion returns correct structure."""
    # Encode a proper assistant response with thinking + end token.
    # The renderer expects tokens as if produced by the model during generation,
    # which means they must end with the stop sequence (<|im_end|> for Qwen3).
    text = "<think>\nLet me think about this.\n</think>\n\nHello, how can I help?<|im_end|>"
    token_ids = qwen_tokenizer.encode(text, add_special_tokens=False)

    result = qwen_tinker_parser.parse_completion(token_ids)

    assert isinstance(result, dict)
    assert "content" in result
    assert "reasoning" in result
    assert "tool_calls" in result
    assert isinstance(result["tool_calls"], list)
    # The thinking should be extracted as reasoning
    assert result["reasoning"]
    assert "think" in result["reasoning"].lower()
    assert "Hello" in result["content"]


def test_tinker_parser_tokenize_and_mask(qwen_tinker_parser):
    """Verify tokenize_and_mask returns correct tensor shapes and mask values."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    prompt_ids, response_ids, response_mask = qwen_tinker_parser.tokenize_and_mask(messages)

    assert prompt_ids.dim() == 1
    assert response_ids.dim() == 1
    assert response_mask.dim() == 1
    assert len(response_ids) == len(response_mask)
    assert len(prompt_ids) > 0
    assert len(response_ids) > 0
    # Response mask should have non-zero values
    assert response_mask.sum() > 0


def test_tinker_parser_tokenize_and_mask_cumulative(qwen_tinker_parser):
    """Verify tokenize_and_mask_cumulative returns correct tensor shapes."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    prompt_ids, response_ids, response_mask = qwen_tinker_parser.tokenize_and_mask_cumulative(messages)

    assert prompt_ids.dim() == 1
    assert response_ids.dim() == 1
    assert response_mask.dim() == 1
    assert len(response_ids) == len(response_mask)
    assert len(prompt_ids) > 0
    assert len(response_ids) > 0
    # Both assistant responses should be masked
    assert response_mask.sum() > 0
    # Should have some zero-masked tokens (user message between assistants)
    assert (response_mask == 0).any()


def test_tinker_parser_verify_equivalence(qwen_tinker_parser):
    """Tinker parser should always return True for verify_equivalence."""
    assert qwen_tinker_parser.verify_equivalence(SIMPLE_TEST_MESSAGES) is True


def test_tinker_parser_matches_manual_qwen(qwen_tokenizer):
    """Compare TinkerChatTemplateParser output with QwenChatTemplateParser for simple messages."""
    renderer = renderers.get_renderer("qwen3", qwen_tokenizer)
    tinker_parser = TinkerChatTemplateParser(renderer)
    manual_parser = QwenChatTemplateParser(qwen_tokenizer)

    # Simple messages without tool calls (avoid tool call format differences)
    simple_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    tinker_result = tinker_parser.parse(simple_messages, add_generation_prompt=False, is_first_msg=True)
    manual_result = manual_parser.parse(simple_messages, add_generation_prompt=False, is_first_msg=True)

    # Tokenize both and compare token sequences (more robust than string comparison
    # because decode round-trip may differ in whitespace/special token rendering).
    # Strip trailing whitespace since HF templates add \n after <|im_end|> but
    # tinker's token-level rendering does not.
    tinker_tokens = qwen_tokenizer.encode(tinker_result.rstrip(), add_special_tokens=False)
    manual_tokens = qwen_tokenizer.encode(manual_result.rstrip(), add_special_tokens=False)
    assert tinker_tokens == manual_tokens


def test_tinker_parser_message_conversion(qwen_tinker_parser):
    """Test that message conversion handles various message formats."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": "Let me search.",
            "tool_calls": [{"function": {"name": "search", "arguments": '{"q": "test"}'}}],
        },
    ]
    converted = qwen_tinker_parser._convert_messages(messages, accumulate_reasoning=False)
    assert len(converted) == 3
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"
    assert converted[2]["role"] == "assistant"


def test_import_error_without_tinker():
    """Verify helpful ImportError when tinker-cookbook is not installed."""
    # Temporarily remove tinker_cookbook from sys.modules
    saved_modules = {}
    modules_to_remove = [key for key in sys.modules if key.startswith("tinker_cookbook")]
    for key in modules_to_remove:
        saved_modules[key] = sys.modules.pop(key)

    try:
        with patch.dict(sys.modules, {"tinker_cookbook": None, "tinker_cookbook.renderers": None}):
            with pytest.raises(ImportError, match="tinker-cookbook is required"):
                from rllm.parser.tinker_parser import _check_tinker_cookbook

                _check_tinker_cookbook()
    finally:
        # Restore modules
        sys.modules.update(saved_modules)
