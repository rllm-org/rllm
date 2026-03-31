import asyncio
import base64
import logging
import os
from io import BytesIO
import json

import openai
from PIL import Image

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool, ToolCall, ToolOutput
from rllm.workflows import TerminationEvent, TerminationReason


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str = "", tokenizer=None, chat_parser=None, max_prompt_length: int = 4096, max_response_length: int = 4096, max_model_length: int | None = None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), sampling_params: dict | None = None, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs):
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = self.sampling_params.pop("reasoning_effort", "medium")

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            # If the caller provides a custom chat parser (e.g. via AgentExecutionEngine),
            # we must use it to ensure consistent tokenization between the execution
            # engine and the rollout engine.
            self.chat_parser = chat_parser or ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            self._use_chat_completions = False
        else:
            # In this case, we cannot enforce max prompt length or dynamically adjust max_tokens <= max_response_length if needed
            print("No tokenizer provided to OpenAIEngine, will use the chat completions endpoint.")
            self._use_chat_completions = True

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _convert_messages_to_openai_format(self, messages: list[dict]) -> list[dict]:
        """Convert messages from rllm format to OpenAI multimodal format."""
        converted_messages = []
        for message in messages:
            if "images" in message and message["images"]:
                content = [{"type": "text", "text": message["content"]}]
                for img in message["images"]:
                    base64_image = self._pil_to_base64(img)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})

                converted_messages.append({"role": message["role"], "content": content})
            else:
                converted_messages.append(message)

        return converted_messages

    def _prepare_max_tokens_param(self, sampling_params: dict, prompt_length: int = None) -> dict:
        """Prepare max tokens parameter for API call (supports O3's max_completion_tokens)."""
        if "max_completion_tokens" in sampling_params:
            return {"max_completion_tokens": sampling_params.pop("max_completion_tokens")}

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        # Adjust for prompt length if provided (completion method needs this)
        if prompt_length and self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining
                print(f"Warning: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return {"max_tokens": max_tokens}

    def _convert_openai_to_tool_calls(self, tool_calls: list[dict] | None) -> list[ToolCall]:
        """Convert OpenAI tool calls to internal ToolCall objects."""
        if not tool_calls:
            return []
        processed_tool_calls: list[ToolCall] = []
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception as e:
                print(f"Error parsing tool call: {tool_call.function.arguments}, error: {e}")
                continue
            processed_tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )
        return processed_tool_calls

    def _convert_tool_calls_to_openai(self, tool_calls: list[ToolCall] | None) -> list[dict] | None:
        """Convert internal ToolCall objects to OpenAI format using base class method."""
        if not tool_calls:
            return None
        return [tool_call.to_openai_format() if isinstance(tool_call, ToolCall) else tool_call for tool_call in tool_calls]

    def _convert_tool_outputs_to_openai(self, tool_outputs: list[ToolOutput] | None) -> list[dict] | None:
        """Convert internal ToolOutput objects to OpenAI format using base class method."""
        if not tool_outputs:
            return None
        return [tool_output.to_openai_format() if isinstance(tool_output, ToolOutput) else tool_output for tool_output in tool_outputs]

    def _prepare_messages_for_openai(self, messages: list[dict]) -> list[dict]:
        """Convert messages from internal format to OpenAI format."""
        openai_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                openai_msg = {"role": "assistant", "content": msg.get("content")}
                if "tool_calls" in msg and msg["tool_calls"]:
                    openai_msg["tool_calls"] = self._convert_tool_calls_to_openai(msg["tool_calls"])
                openai_messages.append(openai_msg)
            elif role == "tool":
                assert "tool_outputs" in msg, "Tool message must contain tool_outputs"
                tool_msgs = self._convert_tool_outputs_to_openai(msg["tool_outputs"])
                if tool_msgs:
                    openai_messages.extend(tool_msgs)
            else:
                openai_messages.append(msg)
        return openai_messages

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("enforce_max_prompt_length", None)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        create_params = self._prepare_max_tokens_param(sampling_params)
        sampling_params.update(create_params)

        tools = sampling_params.pop("tools", self.tools)
        if tools:
            tools = [tool.json if isinstance(tool, Tool) else tool for tool in tools]

        # Convert messages from to OpenAI format
        openai_messages = self._prepare_messages_for_openai(messages)

        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.chat.completions.create(model=self.model, messages=openai_messages, tools=tools, timeout=3600, **sampling_params)
                content = response.choices[0].message.content
                reasoning = response.choices[0].message.reasoning if hasattr(response.choices[0].message, "reasoning") and isinstance(response.choices[0].message.reasoning, str) else ""
                tool_calls = self._convert_openai_to_tool_calls(response.choices[0].message.tool_calls)

                # Build text with reasoning if available, otherwise use content
                if reasoning:
                    text = f"{THOUGHT_DELIMITER_START}\n{reasoning}\n{THOUGHT_DELIMITER_END}\n\n{content}"
                else:
                    text = content

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                return ModelOutput(
                    text=text,
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    prompt_ids=[],
                    completion_ids=[],
                    logprobs=[],
                    prompt_logprobs=[],
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def completion(self, prompt: str | list[int], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if isinstance(prompt, list):
            prompt_ids = prompt
        else:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        create_params = self._prepare_max_tokens_param(sampling_params, prompt_length)
        sampling_params.update(create_params)

        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.completions.create(model=self.model, prompt=prompt, **sampling_params)
                text = response.choices[0].text
                try:
                    completion_ids = response.choices[0].token_ids
                    assert completion_ids is not None
                except Exception:
                    completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

                parsed_output = self.chat_parser.parse_completion(completion_ids)

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                try:
                    assert response.choices[0].logprobs is not None
                    logprobs = response.choices[0].logprobs.token_logprobs
                except Exception:
                    logprobs = []

                if sampling_params.get("echo", False) and logprobs:
                    prompt_logprobs = logprobs[:prompt_length]
                    logprobs = logprobs[prompt_length:]
                elif sampling_params.get("prompt_logprobs", False):
                    try:
                        assert response.choices[0].prompt_logprobs is not None
                        prompt_logprobs: list[float] = [None]
                        for tid, lp in zip(prompt_ids[1:], response.choices[0].prompt_logprobs[1:], strict=False):
                            prompt_logprobs.append(float(lp[str(tid)]["logprob"]))
                    except Exception:
                        prompt_logprobs = []
                else:
                    prompt_logprobs = []

                return ModelOutput(
                    text=text,
                    content=parsed_output["content"],
                    reasoning=parsed_output["reasoning"],
                    tool_calls=parsed_output["tool_calls"],
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        if self._use_chat_completions:
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            if accumulate_reasoning:
                raise ValueError("Accumulate reasoning is not supported for chat completions endpoint.")
            return await self.chat_completion(messages, **kwargs)
        else:
            tools = kwargs.pop("tools", self.tools)
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning, reasoning_effort=reasoning_effort)
            return await self.completion(prompt, **kwargs)

    async def compute_logprobs(self, ids: list[int]) -> list[float]:
        ids = ids[: self.max_model_length]
        output = await self.completion(ids, max_tokens=1, echo=True, logprobs=1, temperature=1.0, top_p=1.0)
        return output.prompt_logprobs
