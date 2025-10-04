import asyncio
import logging
import os

import openai

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser, ToolParser


def parse_openai_error_for_unsupported_param(error_message: str) -> tuple[str | None, str | None]:
    """
    Parse OpenAI API error to extract unsupported parameter and suggested replacement.

    Returns: (unsupported_param, suggested_param) or (None, None) if not parseable

    Example errors:
    - "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
    - "Unsupported value: 'temperature' does not support 0.6 with this model. Only the default (1) value is supported."
    """
    if "unsupported parameter" in error_message.lower():
        # Extract parameter name from quotes
        import re

        match = re.search(r"'([^']+)'\s+is not supported", error_message, re.IGNORECASE)
        if match:
            unsupported = match.group(1)
            # Check for suggested replacement
            suggest_match = re.search(r"use\s+'([^']+)'\s+instead", error_message, re.IGNORECASE)
            suggested = suggest_match.group(1) if suggest_match else None
            return unsupported, suggested

    if "unsupported value" in error_message.lower():
        # Parameter exists but value not allowed - remove the param entirely
        import re

        match = re.search(r"'([^']+)'\s+does not support", error_message, re.IGNORECASE)
        if match:
            return match.group(1), None

    return None, None


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str, tokenizer=None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), sampling_params: dict | None = None, **kwargs):
        self.model = model
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self._param_fixes_logged = set()  # Track which param fixes we've already logged

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            try:
                self.tool_parser = ToolParser.get_parser(self.tokenizer)
            except Exception:
                print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
                self.tool_parser = None
            self._use_chat_completions = False
        else:
            print("No tokenizer provided, will use the chat completions endpoint. This is not recommended.")
            self._use_chat_completions = True

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        sampling_params.pop("model", None)

        retries = self.api_retries
        param_retry_budget = 10  # Allow up to 10 parameter fixes (reasoning models can reject many params)

        while retries > 0:
            try:
                response = await self.client.chat.completions.create(model=self.model, messages=messages, timeout=3600, **sampling_params)
                text = response.choices[0].message.content
                if hasattr(response.choices[0].message, "reasoning") and isinstance(response.choices[0].message.reasoning, str):
                    text = f"{THOUGHT_DELIMITER_START}\n{response.choices[0].message.reasoning}\n{THOUGHT_DELIMITER_END}\n\n{text}"
                return ModelOutput(text=text, tool_calls=response.choices[0].message.tool_calls, finish_reason=response.choices[0].finish_reason, completion_tokens=response.usage.completion_tokens, prompt_tokens=response.usage.prompt_tokens)
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)
            except openai.BadRequestError as e:
                # Try to auto-fix unsupported parameters
                error_msg = str(e)
                unsupported_param, suggested_param = parse_openai_error_for_unsupported_param(error_msg)

                if unsupported_param and param_retry_budget > 0:
                    param_retry_budget -= 1

                    # Only log this fix once per engine instance
                    log_key = f"{unsupported_param}->{suggested_param}" if suggested_param else f"remove:{unsupported_param}"
                    should_log = log_key not in self._param_fixes_logged
                    if should_log:
                        self._param_fixes_logged.add(log_key)
                        print(f"⚠️  Model {self.model} doesn't support '{unsupported_param}', adjusting parameters...")

                    if suggested_param:
                        # Remap parameter (e.g., max_tokens -> max_completion_tokens)
                        if unsupported_param in sampling_params:
                            value = sampling_params.pop(unsupported_param)
                            if suggested_param not in sampling_params:
                                sampling_params[suggested_param] = value
                                if should_log:
                                    print(f"   Remapped '{unsupported_param}' -> '{suggested_param}'")
                    else:
                        # Just remove the unsupported parameter
                        if unsupported_param in sampling_params:
                            sampling_params.pop(unsupported_param)
                            if should_log:
                                print(f"   Removed '{unsupported_param}'")

                    # Retry immediately with fixed params (don't decrement retries)
                    continue

                # Can't auto-fix or out of param retry budget
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def completion(self, prompt: str, **kwargs) -> ModelOutput:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        sampling_params.pop("model", None)

        retries = self.api_retries
        param_retry_budget = 10  # Allow up to 10 parameter fixes (reasoning models can reject many params)

        while retries > 0:
            try:
                response = await self.client.completions.create(model=self.model, prompt=prompt, timeout=3600, **sampling_params)
                return ModelOutput(text=response.choices[0].text, tool_calls=[], finish_reason=response.choices[0].finish_reason, completion_tokens=response.usage.completion_tokens, prompt_tokens=response.usage.prompt_tokens)
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)
            except openai.BadRequestError as e:
                # Try to auto-fix unsupported parameters
                error_msg = str(e)
                unsupported_param, suggested_param = parse_openai_error_for_unsupported_param(error_msg)

                if unsupported_param and param_retry_budget > 0:
                    param_retry_budget -= 1

                    # Only log this fix once per engine instance
                    log_key = f"{unsupported_param}->{suggested_param}" if suggested_param else f"remove:{unsupported_param}"
                    should_log = log_key not in self._param_fixes_logged
                    if should_log:
                        self._param_fixes_logged.add(log_key)
                        print(f"⚠️  Model {self.model} doesn't support '{unsupported_param}', adjusting parameters...")

                    if suggested_param:
                        # Remap parameter (e.g., max_tokens -> max_completion_tokens)
                        if unsupported_param in sampling_params:
                            value = sampling_params.pop(unsupported_param)
                            if suggested_param not in sampling_params:
                                sampling_params[suggested_param] = value
                                if should_log:
                                    print(f"   Remapped '{unsupported_param}' -> '{suggested_param}'")
                    else:
                        # Just remove the unsupported parameter
                        if unsupported_param in sampling_params:
                            sampling_params.pop(unsupported_param)
                            if should_log:
                                print(f"   Removed '{unsupported_param}'")

                    # Retry immediately with fixed params (don't decrement retries)
                    continue

                # Can't auto-fix or out of param retry budget
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)  # only needed for verl engine
        kwargs.pop("validate", None)  # only needed for verl engine
        if self._use_chat_completions:
            return await self.chat_completion(messages, **kwargs)
        else:
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
            output = await self.completion(prompt, **kwargs)
            if self.tool_parser is not None:
                output.tool_calls = self.tool_parser.parse(output.text)
            return output
