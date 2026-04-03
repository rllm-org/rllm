from typing import cast

import tinker
from tinker.types import ModelInput
from tinker_cookbook import model_info, renderers
from typing_extensions import override  # need to use typing_extensions for python < 3.12

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.engine.rollout.types import ImageProcessor, TinkerTokenInput, TinkerTokenOutput, TokenInput, Tokenizer, TokenOutput
from rllm.parser.tinker_parser import TinkerChatTemplateParser
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


def _flat_token_input_length(token_input: TokenInput) -> int:
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

    Wraps the tinker renderer with a TinkerChatTemplateParser, which provides
    unified prompt building (including tool spec injection) and response parsing
    (content, reasoning, tool_calls).
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: Tokenizer,
        service_client: tinker.ServiceClient,
        base_url: str | None = None,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int = 32768,
        sampling_params: dict | None = None,
        image_processor: ImageProcessor | None = None,
        renderer_name: str | None = None,
        **kwargs,
    ):
        """
        Initialize TinkerEngine.

        Args:
            model_name: Name of the model to use
            tokenizer: Tokenizer for encoding/decoding
            service_client: Tinker ServiceClient instance
            base_url: Tinker service URL (default = null for local)
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            max_model_length: Maximum total length (prompt + response) in tokens
            sampling_params: Default sampling parameters (temperature, top_p, etc.)
            image_processor: Optional image processor for vision-language models (used with renderer)
            renderer_name: Optional renderer name to use (None = auto-detect from model)
            kwargs: Additional keyword arguments
                - strip_thinking_from_history: Whether to strip thinking from history (only for Qwen3Renderer)
        """
        super().__init__()
        self.base_url = base_url
        self.model_name = model_name
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1
        self.tokenizer = tokenizer

        self.train_sampling_params = dict(sampling_params.get("train", {})) if sampling_params else {}
        self.val_sampling_params = dict(sampling_params.get("val", {})) if sampling_params else {}
        # Initialize Tinker service client
        self.service_client = service_client

        # Initialize the renderer and wrap with TinkerChatTemplateParser
        renderer_name = renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = renderers.get_renderer(renderer_name, self.tokenizer, image_processor=image_processor)

        if "strip_thinking_from_history" in kwargs and isinstance(kwargs["strip_thinking_from_history"], bool) and hasattr(renderer, "strip_thinking_from_history"):
            renderer.strip_thinking_from_history = kwargs["strip_thinking_from_history"]

        self.chat_parser: TinkerChatTemplateParser = TinkerChatTemplateParser(renderer)
        self.stop_sequences = self.chat_parser.stop_sequences

        # Sampling client will be set via set_sampling_client()
        self.sampling_client: tinker.SamplingClient | None = None

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """
        Set the sampling client for inference.

        Args:
            sampling_client: Tinker SamplingClient instance
        """
        self.sampling_client = sampling_client

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

        # prepare sampling params
        sampling_params = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()

        requested_max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self.max_response_length))
        requested_max_tokens = sampling_params.pop("max_tokens", requested_max_tokens)
        max_tokens = self._prepare_max_tokens(requested_max_tokens, input_length)

        if "temperature" in kwargs:
            sampling_params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            sampling_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            sampling_params["top_k"] = kwargs["top_k"]

        tinker_sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            stop=self.stop_sequences,  # type: ignore
            **sampling_params,
        )
        # call sampling client
        model_input = _flat_token_input_to_model_input(token_input)
        sample_response: tinker.SampleResponse = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker_sampling_params,
        )

        # return sampled sequence from sample response
        return sample_response.sequences[0]

    @override
    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput) -> ModelOutput:
        """
        Assemble model output from a sampled sequence.
        """
        sampled_sequence = cast(TinkerTokenOutput, token_output)
        response_tokens, logprobs = sampled_sequence.tokens, sampled_sequence.logprobs

        # Parse response using parser (handles content, reasoning, tool_calls)
        parsed_output = self.chat_parser.parse_completion(response_tokens)
        content = parsed_output.get("content", "")
        reasoning = parsed_output.get("reasoning", "")
        tool_calls = parsed_output.get("tool_calls", [])

        # decode full text
        completion_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)  # type: ignore
        finish_reason = sampled_sequence.stop_reason
        # special handling for prompt ids, we will break any EncodedTextChunk into ints
        prompt_ids = []
        for elem in token_input:
            if isinstance(elem, tinker.EncodedTextChunk):
                prompt_ids.extend(elem.tokens)
            else:
                prompt_ids.append(elem)

        return ModelOutput(
            text=completion_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=response_tokens,
            logprobs=logprobs,
            routing_matrices=None,
            prompt_length=_flat_token_input_length(token_input),
            completion_length=len(response_tokens),
            finish_reason=finish_reason,
            metrics=None,
        )

    @override
    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """
        Generate model response for a given set of messages.

        Args:
            messages: List of message dictionaries (OpenAI format)
            **kwargs: Additional parameters including:
                - application_id: Session/application ID for tracing
                - enforce_max_prompt_length: Whether to enforce max prompt length
                - tools: List of tools for tool-augmented generation

        Returns:
            ModelOutput with generated text and metadata
        """
        # Extract unused kwargs
        kwargs.pop("application_id", None)

        # Extract tools
        tools = kwargs.pop("tools", [])

        # Build prompt using TinkerChatTemplateParser (handles tools, images, etc.)
        tinker_prompt = self.chat_parser.build_prompt(messages, tools=tools)
        token_input: TinkerTokenInput = tinker_prompt.chunks

        sampled_sequence = await self.get_token_output_from_token_input(token_input=token_input, **kwargs)
        return self.assemble_model_output(token_input=token_input, token_output=sampled_sequence)

    async def compute_logprobs(self, ids: list[int]) -> list[float]:
        ids = ids[: self.max_model_length]
        return await self.sampling_client.compute_logprobs_async(ModelInput.from_ints(ids))
