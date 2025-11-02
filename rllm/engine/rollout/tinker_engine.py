import tinker
from tinker_cookbook import model_info, renderers

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.workflows import TerminationEvent, TerminationReason


class TinkerEngine(RolloutEngine):
    """
    RolloutEngine implementation using Tinker for model inference.

    Uses Tinker's renderer system for response parsing instead of ChatTemplateParser.
    """

    def __init__(self, base_url: str, model_name: str, tokenizer, service_client: tinker.ServiceClient, max_prompt_length: int = 4096, max_response_length: int = 4096, sampling_params: dict | None = None, **kwargs):
        """
        Initialize TinkerEngine.

        Args:
            base_url: Tinker service base URL
            model_name: Name of the model to use
            tokenizer: Tokenizer for encoding/decoding
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            sampling_params: Default sampling parameters (temperature, top_p, etc.)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.tokenizer = tokenizer
        self.default_sampling_params = sampling_params or {}

        # Initialize Tinker service client
        self.service_client = service_client

        # Initialize renderer using model info
        renderer_name = model_info.get_recommended_renderer_name(self.model_name)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)

        # Set up sampling parameters
        self.sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_response_length,
            stop=self.renderer.get_stop_sequences(),
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

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """
        Generate model response for a given set of messages.

        Args:
            messages: List of message dictionaries (OpenAI format)
            **kwargs: Additional parameters including:
                - application_id: Session/application ID for tracing
                - validate: Whether this is validation (for greedy decoding)
                - enforce_max_prompt_length: Whether to enforce max prompt length

        Returns:
            ModelOutput with generated text and metadata
        """
        if self.sampling_client is None:
            raise RuntimeError("Sampling client not set. Call set_sampling_client() first.")

        # Extract kwargs
        kwargs.pop("application_id", None)
        kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # Prepare sampling params (override defaults with kwargs)
        sampling_params = tinker.types.SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.max_response_length),
            stop=self.renderer.get_stop_sequences(),
            temperature=kwargs.get("temperature", self.default_sampling_params.get("temperature", 1.0)),
            top_p=kwargs.get("top_p", self.default_sampling_params.get("top_p", 1.0)),
        )

        # Build prompt using renderer (converts messages to Tinker prompt)
        tinker_prompt = self.renderer.build_generation_prompt(messages)
        prompt_ids = tinker_prompt.to_ints()
        prompt_length = len(prompt_ids)

        # Check prompt length
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

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
