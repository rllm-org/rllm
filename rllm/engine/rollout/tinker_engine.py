from typing import cast

import tinker
from tinker.types import ModelInput
from tinker_cookbook import model_info, renderers
from typing_extensions import override  # need to use typing_extensions for python < 3.12

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine, TinkerTokenInput, TinkerTokenOutput, TokenInput
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

"""
Utility functions for Tinker engine. Partly borrowed from
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
"""


def _flat_token_input_to_model_input(token_input: TinkerTokenInput) -> ModelInput:
    """Convert a flat token input to a ModelInput."""
    if not token_input:  # empty list
        return ModelInput(chunks=[])

    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in token_input:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)

    flush_text_chunk()  # final clear up
    return tinker.ModelInput(chunks=out)


def _flat_token_input_length(token_input: TinkerTokenInput) -> int:
    """Get the length of a flat token input. This nicely handles both text and image inputs"""
    length = 0
    for elem in token_input:
        if isinstance(elem, int):
            length += 1
        else:
            length += elem.length
    return length


class TinkerEngine(RolloutEngine):
    """
    RolloutEngine implementation using Tinker for model inference.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        tokenizer,
        service_client: tinker.ServiceClient,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        bypass_render_with_parser: bool = False,
        processor=None,
        image_processor=None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        """
        Initialize TinkerEngine.

        Args:
            base_url: Tinker service base URL
            model_name: Name of the model to use
            tokenizer: Tokenizer for encoding/decoding
            service_client: Tinker ServiceClient instance
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            max_model_length: Maximum total length (prompt + response) in tokens
            sampling_params: Default sampling parameters (temperature, top_p, etc.)
            bypass_render_with_parser: If True, use ChatTemplateParser instead of Tinker's renderer
            processor: Optional processor for multimodal models (used when bypass_render_with_parser=True)
            image_processor: Optional image processor for vision-language models (used with renderer)
            disable_thinking: Whether to disable thinking in generation prompt (used when bypass_render_with_parser=True)
            accumulate_reasoning: Whether to accumulate reasoning (used when bypass_render_with_parser=True)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.tokenizer = tokenizer
        self.default_sampling_params = sampling_params or {}
        self.bypass_render_with_parser = bypass_render_with_parser
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        # Initialize Tinker service client
        self.service_client = service_client

        if bypass_render_with_parser:
            self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=disable_thinking)
            self.renderer = None
            if hasattr(self.chat_parser, "stop_sequences") and self.chat_parser.stop_sequences:
                self.stop_sequences = self.chat_parser.stop_sequences
            elif hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
                self.stop_sequences = [tokenizer.eos_token]
            else:
                raise ValueError("No stop sequences found for tokenizer or chat parser")
        else:
            renderer_name = model_info.get_recommended_renderer_name(self.model_name)
            # Pass image_processor for VLM support with Tinker renderer
            self.renderer = renderers.get_renderer(renderer_name, self.tokenizer, image_processor=image_processor)
            self.chat_parser = None
            self.stop_sequences = self.renderer.get_stop_sequences()

        # Sampling client will be set via set_sampling_client()
        self.sampling_client = None

    def set_sampling_client(self, sampling_client):
        """
        Set the sampling client for inference.

        Args:
            sampling_client: Tinker SamplingClient instance
        """
        self.sampling_client = sampling_client

    def _convert_images_to_content_list(self, messages: list[dict]) -> list[dict]:
        """
        Convert messages from standard format to Tinker renderer format.

        Standard format: {"role": "user", "content": "text", "images": [PIL.Image]}
        Tinker format:   {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "..."}]}

        Args:
            messages: List of messages in standard format

        Returns:
            List of messages in Tinker renderer format
        """
        converted = []
        for msg in messages:
            if "images" in msg and msg["images"]:
                # Convert to content list format
                content_list = []
                for img in msg["images"]:
                    content_list.append({"type": "image", "image": img})
                content_list.append({"type": "text", "text": msg.get("content", "")})
                converted.append({**msg, "content": content_list})
                # Remove the images key since it's now in content
                del converted[-1]["images"]
            else:
                converted.append(msg)
        return converted

    def _prepare_max_tokens(self, requested_max_tokens: int, prompt_length: int) -> int:
        """
        Prepare max_tokens parameter, adjusting for max_model_length if needed.

        Args:
            requested_max_tokens: The requested max_tokens value
            prompt_length: The length of the prompt in tokens

        Returns:
            Adjusted max_tokens value
        """
        max_tokens = requested_max_tokens

        # Adjust for prompt length if max_model_length is set
        if self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining
                print(f"Warning: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return max_tokens

    @property
    def supports_token_in_token_out(self) -> bool:
        """Tinker sampling client does support returning prompt_ids, so this is true."""
        return True

    @override
    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> TinkerTokenOutput:
        """
        Generate a sampled sequence from a given token input.
        """
        token_input = cast(TinkerTokenInput, token_input)
        if self.sampling_client is None:
            raise RuntimeError("Sampling client not set. Call set_sampling_client() first.")

        input_length = _flat_token_input_length(token_input)

        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        if enforce_max_prompt_length and input_length > min(self.max_prompt_length, self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        requested_max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_response_length))
        max_tokens = self._prepare_max_tokens(requested_max_tokens, input_length)

        # prepare sampling params
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            stop=self.stop_sequences,  # type: ignore
            temperature=kwargs.get("temperature", self.default_sampling_params.get("temperature", 1.0)),
            top_p=kwargs.get("top_p", self.default_sampling_params.get("top_p", 1.0)),
        )

        # call sampling client
        model_input = _flat_token_input_to_model_input(token_input)
        sample_response: tinker.SampleResponse = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        # return sampled sequence from sample response
        return sample_response.sequences[0]

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """
        Generate model response for a given set of messages.

        Args:
            messages: List of message dictionaries (OpenAI format)
            **kwargs: Additional parameters including:
                - application_id: Session/application ID for tracing
                - validate: Whether this is validation (for greedy decoding)
                - enforce_max_prompt_length: Whether to enforce max prompt length
                - tools: List of tools (used when bypass_render_with_parser=True)
                - accumulate_reasoning: Whether to accumulate reasoning (used when bypass_render_with_parser=True)

        Returns:
            ModelOutput with generated text and metadata
        """
        # Extract unused kwargs
        kwargs.pop("application_id", None)
        kwargs.pop("validate", False)

        # Extract parser-specific kwargs
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        if self.bypass_render_with_parser:
            # Use ChatTemplateParser
            prompt = self.chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=tools,
                reasoning_effort=reasoning_effort,
                accumulate_reasoning=accumulate_reasoning,
            )
            token_input: list[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_length = len(token_input)
        else:
            # Use Tinker renderer
            # Convert standard image format to Tinker renderer format
            converted_messages = self._convert_images_to_content_list(messages)
            # Build prompt using renderer
            token_input: TinkerTokenInput = self.renderer.build_generation_prompt(converted_messages).chunks  # type: ignore
            prompt_length = _flat_token_input_length(token_input)

        sampled_sequence = await self.get_token_output_from_token_input(token_input=token_input, **kwargs.copy())
        response_tokens = sampled_sequence.tokens
        logprobs = sampled_sequence.logprobs

        # Parse response using renderer
        response_dict, _ = self.renderer.parse_response(response_tokens)

        # Extract content from response
        if isinstance(response_dict, dict):
            content = response_dict.get("content", "")
            reasoning = response_dict.get("reasoning", "")
            tool_calls = response_dict.get("tool_calls", [])
        else:
            content = response_dict if isinstance(response_dict, str) else ""
            reasoning = ""
            tool_calls = []

        # Decode full text
        completion_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Determine finish reason
        requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_response_length))
        max_tokens = self._prepare_max_tokens(requested_max_tokens, prompt_length) if prompt_length > 0 else requested_max_tokens
        finish_reason = "length" if len(response_tokens) >= max_tokens else "stop"

        return ModelOutput(
            text=completion_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=token_input,
            completion_ids=response_tokens,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(response_tokens),
            finish_reason=finish_reason,
        )
