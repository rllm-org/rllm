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
    assert "<think>\\n\\n</think>\\n\\n" in parser.assistant_token

    # Test equivalence check
    assert parser.verify_equivalence(PARSER_TEST_MESSAGES)
