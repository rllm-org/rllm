import logging
from datetime import datetime

import requests
import torch

from .utils import PARSER_TEST_MESSAGES, fix_pad_token

logger = logging.getLogger(__name__)


class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""
        self.generation_prompt_ids = self._get_generation_prompt_ids(tokenizer)

        # Fix pad_token if it's the same as eos_token
        fix_pad_token(self.tokenizer)

    def _get_generation_prompt_ids(self, tokenizer):
        """Return the generation prompt tokens (ids, tokens, decoded string)."""
        messages = [{"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")

        with_ids = with_prompt[0].tolist()
        without_ids = without_prompt[0].tolist()

        generation_prompt_ids = with_ids[len(without_ids) :]

        return generation_prompt_ids

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

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
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
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
    def get_parser(cls, tokenizer, disable_thinking=False) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)
            elif "harmony" in model_name or "gpt-oss" in model_name:
                logger.info(f"Using HarmonyChatTemplateParser for {tokenizer.name_or_path}")
                return HarmonyChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages, mask_last_assistant_only=False):
        prompt_ids = []
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        for i in range(first_assistant_idx):
            parsed_msg = self.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            prompt_ids.extend(ids)

        for i in range(first_assistant_idx, len(messages)):
            parsed_msg = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            response_ids.extend(ids)

            if messages[i]["role"] == "assistant":
                # For assistant messages, response_mask should be 1 for all tokens except the generation prompt, which should be 0
                if ids[: len(self.generation_prompt_ids)] != self.generation_prompt_ids:
                    logger.warning(f"Generation prompt mismatch for message {i}\nexpected generation_prompt_ids: {self.generation_prompt_ids}\nactual_ids: {ids[: len(self.generation_prompt_ids)]}\nexpected generation_prompt: {self.tokenizer.decode(self.generation_prompt_ids, skip_special_tokens=False)}\nactual prompt: {self.tokenizer.decode(ids[: len(self.generation_prompt_ids)], skip_special_tokens=False)}")

                num_non_gen_prompt = len(ids) - len(self.generation_prompt_ids)

                if mask_last_assistant_only and i != last_assistant_idx:
                    response_mask.extend([0] * len(ids))
                else:
                    response_mask.extend([0] * len(self.generation_prompt_ids))
                    response_mask.extend([1] * num_non_gen_prompt)
            else:
                response_mask.extend([0] * len(ids))

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.eos_token + self.assistant_token + "<think>\n"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eos_token


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\\n\\n</think>\\n\\n"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"

        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # took tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token


class HarmonyChatTemplateParser(ChatTemplateParser):
    """Parser for OpenAI Harmony format used by gpt-oss models.

    The Harmony format supports 5 roles: system, developer, user, assistant, tool
    Assistant messages can have channels: final, analysis, commentary
    Tool calls use recipients and constraints for structured interactions.
    """

    def __init__(self, tokenizer, default_channel="final", reasoning_effort="high"):
        super().__init__(tokenizer)
        self.start_token = "<|start|>"
        self.end_token = "<|end|>"
        self.message_token = "<|message|>"
        self.channel_token = "<|channel|>"
        self.constrain_token = "<|constrain|>"
        self.call_token = "<|call|>"
        self.default_channel = default_channel
        self.reasoning_effort = reasoning_effort
        assert self.reasoning_effort in ["low", "medium", "high"], f"Invalid reasoning effort: {self.reasoning_effort}"

        self._cached_date = self._get_current_date()

    def _get_current_date(self):
        """Get current date dynamically from API or fallback to system date."""
        try:
            # Try to get date from worldtimeapi.org
            response = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC", timeout=2)
            if response.status_code == 200:
                data = response.json()
                date_str = data["datetime"][:10]  # Extract YYYY-MM-DD
                return date_str
        except Exception:
            pass

        try:
            # Fallback to timeapi.io
            response = requests.get("http://timeapi.io/api/Time/current/zone?timeZone=UTC", timeout=2)
            if response.status_code == 200:
                data = response.json()
                date_str = data.get("date", "")
                # Handle different date formats
                try:
                    # Try MM/DD/YYYY format
                    if "/" in date_str:
                        date_obj = datetime.strptime(date_str, "%m/%d/%Y")
                        return date_obj.strftime("%Y-%m-%d")
                    # Try YYYY-MM-DD format
                    elif "-" in date_str and len(date_str) == 10:
                        datetime.strptime(date_str, "%Y-%m-%d")  # Validate format
                        return date_str
                except ValueError:
                    pass
        except Exception:
            pass

        try:
            # Try another simple API
            response = requests.get("http://date.jsontest.com/", timeout=2)
            if response.status_code == 200:
                data = response.json()
                date_str = data.get("date", "")
                # Usually returns MM-DD-YYYY format
                if "-" in date_str:
                    try:
                        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
        except Exception:
            pass

        # Final fallback to system date
        return datetime.now().strftime("%Y-%m-%d")

    def _get_default_system_message(self):
        """Get the default system message with cached current date."""
        current_date = self._cached_date
        return f"""You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: {self.reasoning_effort}

# Valid channels: analysis, commentary, final. Channel must be included for every message."""

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        # Always add default system message first when it's the first message
        if is_first_msg:
            result += self._format_message("system", self._get_default_system_message())
            result += self._format_message("developer", "")

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "developer":
                result += self.parse_developer(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def _format_message(self, role, content, channel=None, recipient=None, constraint=None, is_call=False):
        """Helper method to format messages according to Harmony format."""
        result = self.start_token + role

        if recipient:
            result += f" to={recipient}"

        if channel:
            result += self.channel_token + channel

        if constraint:
            result += f" {self.constrain_token}{constraint}"

        result += self.message_token + content

        if is_call:
            result += self.call_token
        else:
            result += self.end_token

        return result

    def parse_system(self, message):
        return self._format_message("system", message["content"])

    def parse_developer(self, message):
        return self._format_message("developer", message["content"])

    def parse_user(self, message):
        return self._format_message("user", message["content"])

    def parse_assistant(self, message):
        # Extract harmony-specific metadata from message
        channel = message.get("channel", self.default_channel)
        recipient = message.get("recipient")
        constraint = message.get("constraint")
        is_call = message.get("is_call", False)

        return self._format_message("assistant", message["content"], channel=channel, recipient=recipient, constraint=constraint, is_call=is_call)

    def parse_tool(self, message):
        # Tool messages format: <|start|>{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>
        tool_name = message.get("name", "tool")
        channel = message.get("channel", "commentary")
        recipient = message.get("recipient", "assistant")

        return self._format_message(tool_name, message["content"], channel=channel, recipient=recipient)

    @property
    def generation_prompt(self):
        """Generate the prompt to start assistant generation."""
        return f"{self.start_token}assistant{self.channel_token}{self.default_channel}{self.message_token}"
