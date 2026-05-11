"""Base class for chat-template parsers.

Each per-family parser (Qwen, DeepseekQwen, Llama, ...) lives in its own
module under ``rllm.parser.chat_template``. Importing ``ChatTemplateParser``
from ``rllm.parser`` continues to work via the re-export shim in
``rllm/parser/chat_template_parser.py``.
"""

from __future__ import annotations

import logging

from rllm.parser.messages import Messages
from rllm.parser.utils import PARSER_TEST_MESSAGES

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch

        return torch
    except ImportError as err:
        raise ImportError("ChatTemplateParser.tokenize_and_mask requires PyTorch. Install with: pip install rllm[train]") from err


class ChatTemplateParser:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_prompt = self._get_generation_prompt(tokenizer)

    def _get_generation_prompt(self, tokenizer):
        # Some chat templates (e.g. Qwen3.5) reject a lone assistant message,
        # so prepend a stub user message. It is present in both with_prompt
        # and without_prompt and cancels out in the slice below.
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        generation_prompt = with_prompt[len(without_prompt) :]

        return generation_prompt

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, **kwargs) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def parse_completion(self, completion_ids: list[int]):
        raise NotImplementedError("ChatTemplateParser does not support parse_completion")

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        batch_result = self.parse(messages)

        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, processor=None, disable_thinking=False) -> ChatTemplateParser:
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser
            processor: Optional HF processor for multimodal models
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Lazy imports to avoid circular dependencies across the chat_template package.
        from rllm.parser.chat_template.deepseek_qwen import DeepseekQwenChatTemplateParser
        from rllm.parser.chat_template.deepseek_v32_exp import DeepSeekV32ExpChatTemplateParser
        from rllm.parser.chat_template.harmony import HarmonyChatTemplateParser
        from rllm.parser.chat_template.kimi_k2_thinking import KimiK2ThinkingChatTemplateParser
        from rllm.parser.chat_template.llama import LlamaChatTemplateParser
        from rllm.parser.chat_template.qwen import QwenChatTemplateParser

        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and ("llama" in tokenizer_cls or "distill-qwen" in model_name):
                if "deepseek-math-v2" in model_name or "deepseek-v3.2-exp" in model_name:
                    logger.info(f"Using DeepSeekV32ExpChatTemplateParser for {tokenizer.name_or_path}")
                    return DeepSeekV32ExpChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
                else:
                    logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                    return DeepseekQwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, processor=processor, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)
            elif "gpt-oss" in model_name or "imo" in model_name:
                logger.info(f"Using HarmonyChatTemplateParser for {tokenizer.name_or_path}")
                return HarmonyChatTemplateParser()
            elif "kimi-k2" in model_name:
                logger.info(f"Using KimiK2ThinkingChatTemplateParser for {tokenizer.name_or_path}")
                return KimiK2ThinkingChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer, processor=processor)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages):
        try:
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:last_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response = self.parse([messages[last_assistant_idx]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
        response = response[len(self.generation_prompt) :].rstrip("\n")  # handle qwen trailing newline from eot token
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_mask = [1] * len(response_ids)

        torch = _import_torch()
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages):
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            if is_asst:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                response = response[len(self.generation_prompt) :]
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([1] * len(ids))
            else:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([0] * len(ids))

        torch = _import_torch()
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask
