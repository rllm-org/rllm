from transformers import AutoTokenizer

from rllm.parser import (
    ChatTemplateParser,
    DeepseekQwenChatTemplateParser,
    LlamaChatTemplateParser,
    QwenChatTemplateParser,
)
from rllm.parser.utils import PARSER_TEST_MESSAGES


def test_qwen_chat_template_parser():
    # Test with Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    parser = QwenChatTemplateParser(tokenizer)

    # Test equivalence check
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)

    # Test parsing with generation prompt
    result = parser.parse(PARSER_TEST_MESSAGES, add_generation_prompt=True)
    assert isinstance(result, str)
    assert len(result) > 0
    assert parser.assistant_token in result


def test_deepseek_qwen_chat_template_parser():
    # Test with Deepseek-Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser = DeepseekQwenChatTemplateParser(tokenizer)

    # Test equivalence check
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)

    # Test basic parsing
    result = parser.parse(PARSER_TEST_MESSAGES)
    assert isinstance(result, str)
    assert len(result) > 0


def test_llama_chat_template_parser():
    # Use a public Llama model instead of gated Meta-Llama
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser = LlamaChatTemplateParser(tokenizer)

    # Test equivalence check
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)

    # Test basic parsing
    result = parser.parse(PARSER_TEST_MESSAGES)
    assert isinstance(result, str)
    assert len(result) > 0
    assert parser.assistant_token in result


def test_parser_factory():
    # Test Qwen model
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    qwen_parser = ChatTemplateParser.get_parser(qwen_tokenizer)
    assert isinstance(qwen_parser, QwenChatTemplateParser)
    assert qwen_parser.verify_equivalence(PARSER_TEST_MESSAGES)

    # Test Deepseek-Qwen model
    deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_parser = ChatTemplateParser.get_parser(deepseek_tokenizer)
    assert isinstance(deepseek_parser, DeepseekQwenChatTemplateParser)
    assert deepseek_parser.verify_equivalence(PARSER_TEST_MESSAGES)


def test_parser_with_disable_thinking():
    # Test Qwen parser with thinking disabled
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=True)

    # Verify that thinking is disabled in the generation prompt
    assert "<think>\n\n</think>\n\n" in parser.assistant_token

    # Test equivalence check
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)


# Mimics the strict alternation rule used by some newer chat templates
# (e.g. Qwen3.5): the first message must be `system` or `user`.
_STRICT_ALTERNATION_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if loop.index0 == 0 and message['role'] not in ['system', 'user'] -%}"
    "{{ raise_exception('First message must be system or user') }}"
    "{%- endif -%}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
)


def test_get_generation_prompt_handles_strict_alternation_templates():
    """Regression test for Qwen3.5-style chat templates."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tokenizer.chat_template = _STRICT_ALTERNATION_TEMPLATE

    parser = ChatTemplateParser(tokenizer)
    assert parser.generation_prompt
    assert "assistant" in parser.generation_prompt


def test_qwen3_5_chat_template_parser():
    """End-to-end test on the actual Qwen3.5 tokenizer."""
    import pytest

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    except Exception as e:
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")

    parser = ChatTemplateParser.get_parser(tokenizer)
    assert isinstance(parser, QwenChatTemplateParser)
    assert parser.generation_prompt
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)


def test_chat_parser_satisfies_base_parser():
    """ChatTemplateParser must honor the BaseParser contract so it is
    interchangeable with RendererParser inside the rollout engines."""
    from rllm.parser import BaseParser
    from rllm.parser.base import ParsedCompletion

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    parser = ChatTemplateParser.get_parser(tokenizer)
    assert isinstance(parser, BaseParser)

    # render -> token ids
    token_ids = parser.render(PARSER_TEST_MESSAGES, add_generation_prompt=True)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0 and all(isinstance(t, int) for t in token_ids)

    # get_stop_token_ids -> non-empty list[int]
    stop_ids = parser.get_stop_token_ids()
    assert stop_ids and all(isinstance(t, int) for t in stop_ids)

    # parse_completion -> ParsedCompletion, with dict-style back-compat access
    completion = tokenizer.encode("<think>reasoning</think>answer", add_special_tokens=False)
    parsed = parser.parse_completion(completion)
    assert isinstance(parsed, ParsedCompletion)
    assert parsed.content == "answer" == parsed["content"]
    assert parsed.reasoning == "reasoning" == parsed.get("reasoning")
