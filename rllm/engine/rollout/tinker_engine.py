import tinker
from tinker.types import ModelInput
from tinker_cookbook import model_info, renderers

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


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

        # Set up sampling parameters
        self.sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_response_length,
            stop=self.stop_sequences,
            temperature=self.default_sampling_params.get("temperature", 1.0),
            top_p=self.default_sampling_params.get("top_p", 1.0),
        )

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
        if self.sampling_client is None:
            raise RuntimeError("Sampling client not set. Call set_sampling_client() first.")

        # Extract kwargs
        kwargs.pop("application_id", None)
        kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

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
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_length = len(prompt_ids)

            # Check prompt length
            if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

            # Dynamically adjust max_tokens based on prompt length
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_response_length))
            max_tokens = self._prepare_max_tokens(requested_max_tokens, prompt_length)

            # Prepare sampling params (override defaults with kwargs)
            sampling_params = tinker.types.SamplingParams(
                max_tokens=max_tokens,
                stop=self.stop_sequences,
                temperature=kwargs.get("temperature", self.default_sampling_params.get("temperature", 1.0)),
                top_p=kwargs.get("top_p", self.default_sampling_params.get("top_p", 1.0)),
            )

            # Convert prompt to Tinker prompt format
            tinker_prompt = ModelInput.from_ints(prompt_ids)

            # Call Tinker sampling API
            sample_response = await self.sampling_client.sample_async(
                prompt=tinker_prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )

            # Extract response tokens and logprobs
            response_tokens = sample_response.sequences[0].tokens
            logprobs = sample_response.sequences[0].logprobs

            # Parse response using parser
            parsed_output = self.chat_parser.parse_completion(response_tokens)

            content = parsed_output.get("content", "")
            reasoning = parsed_output.get("reasoning", "")
            tool_calls = parsed_output.get("tool_calls", [])

            # Decode full text
            completion_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            # Use Tinker renderer (original behavior)
            # Convert standard image format to Tinker renderer format
            converted_messages = self._convert_images_to_content_list(messages)
            # Build prompt using renderer (converts messages to Tinker prompt)
            tinker_prompt = self.renderer.build_generation_prompt(converted_messages)

            # For VLM prompts with ImageChunks, to_ints() may not be supported
            try:
                prompt_ids = tinker_prompt.to_ints()
                prompt_length = len(prompt_ids)
            except ValueError:
                # Prompt contains ImageChunks - skip length enforcement for VLM
                prompt_ids = []
                prompt_length = 0

            # Check prompt length (only for text-only prompts)
            if prompt_length > 0 and enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

            # Dynamically adjust max_tokens based on prompt length
            requested_max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_response_length))
            max_tokens = self._prepare_max_tokens(requested_max_tokens, prompt_length) if prompt_length > 0 else requested_max_tokens

            # Prepare sampling params (override defaults with kwargs)
            sampling_params = tinker.types.SamplingParams(
                max_tokens=max_tokens,
                stop=self.stop_sequences,
                temperature=kwargs.get("temperature", self.default_sampling_params.get("temperature", 1.0)),
                top_p=kwargs.get("top_p", self.default_sampling_params.get("top_p", 1.0)),
            )

            # Call Tinker sampling API
            sample_response = await self.sampling_client.sample_async(
                prompt=tinker_prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )

            # Extract response tokens and logprobs
            response_tokens = sample_response.sequences[0].tokens
            logprobs = sample_response.sequences[0].logprobs

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
        finish_reason = "stop"
        if len(response_tokens) >= sampling_params.max_tokens:
            finish_reason = "length"

        return ModelOutput(
            text=completion_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=response_tokens,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(response_tokens),
            finish_reason=finish_reason,
        )
