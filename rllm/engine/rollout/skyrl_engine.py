"""
SkyRL integration for rLLM:
SkyRLEngine: Adapts SkyRL's InferenceEngineClient to rLLM's RolloutEngine interface.

This adapter allows rLLM workflows to use SkyRL's inference backends
(vLLM, SGLang, etc.) transparently during trajectory generation.
"""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add skyrl-train to Python path
skyrl_train_path = Path(__file__).parent.parent.parent.parent / "skyrl" / "skyrl-train"
if skyrl_train_path.exists():
    sys.path.insert(0, str(skyrl_train_path))

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class SkyRLEngine(RolloutEngine):
    """Adapts SkyRL's InferenceEngineClient to rLLM's RolloutEngine interface.

    This adapter allows rLLM workflows to use SkyRL's inference backends
    (vLLM, SGLang, etc.) transparently during trajectory generation.
    """

    def __init__(
        self,
        inference_engine_client: InferenceEngineClient | None = None,
        tokenizer=None,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        config=None,
        **kwargs,
    ):
        """Initialize the wrapper.

        Args:
            inference_engine_client: SkyRL's InferenceEngineClient (optional, can be set later)
            tokenizer: Tokenizer instance
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
            config: Configuration object (optional, for backward compatibility)
                If provided, can extract max_prompt_length and max_response_length from config.data
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Extract max lengths from config if provided
        if config is not None:
            if hasattr(config, "data"):
                max_prompt_length = config.data.get("max_prompt_length", max_prompt_length)
                max_response_length = config.data.get("max_response_length", max_response_length)
        
        self.inference_engine = inference_engine_client
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.config = config
        self.skyrl_trainer = None  # Can be set later via set_skyrl_components
        self.validate = False  # Flag enabled/disabled by SkyRLBackend validation hooks

        # Add ChatTemplateParser (matching VerlEngine pattern)
        disable_thinking = config.get("rllm", {}).get("disable_thinking", False) if config else False
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            processor=None,
            disable_thinking=disable_thinking,
        )

    def set_skyrl_components(
        self,
        inference_engine_client: InferenceEngineClient | None = None,
        trainer=None,
    ):
        """Set SkyRL components after initialization.

        Args:
            inference_engine_client: SkyRL InferenceEngineClient
            trainer: SkyRL RayPPOTrainer
        """
        if inference_engine_client is not None:
            self.inference_engine = inference_engine_client
        if trainer is not None:
            self.skyrl_trainer = trainer

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Get model response using SkyRL's inference engine.

        Args:
            messages: List of chat messages in OpenAI format
            **kwargs: Additional parameters including:
                - sampling_params: Dict with temperature, top_p, max_tokens, etc.
                - validate: Whether this is validation (for greedy decoding)
                - enforce_max_prompt_length: Whether to enforce max prompt length

        Returns:
            ModelOutput: Structured model response with token IDs and metadata
        """
        from skyrl_train.inference_engines.base import InferenceEngineInput
        from rllm.workflows import TerminationEvent, TerminationReason

        if self.inference_engine is None:
            raise RuntimeError("InferenceEngineClient not set. Call set_skyrl_components() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        # Extract parameters
        sampling_params = kwargs.get("sampling_params", {})
        validate = self.validate or kwargs.get("validate", False)
        enforce_max_prompt_length = kwargs.get("enforce_max_prompt_length", True)

        # Extract kwargs for parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", False)

        # Use chat_parser for proper formatting (matching VerlEngine)
        prompt_text = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        # Enforce prompt length limit
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Prepare SkyRL inference input
        inference_input: InferenceEngineInput = {
            "prompts": [messages],
            "prompt_token_ids": None,
            "sampling_params": {
                "max_tokens": sampling_params.get("max_tokens", self.max_response_length),
                "temperature": 0.0 if validate else sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                **{k: v for k, v in sampling_params.items() if k not in ["max_tokens", "temperature", "top_p"]}
            },
            "session_ids": None,
        }

        # Call SkyRL's inference engine
        output = await self.inference_engine.generate(inference_input)

        # Extract response
        response_text = output["responses"][0]
        response_ids = output["response_ids"][0]
        stop_reason = output["stop_reasons"][0]
        logprobs = output.get("response_logprobs", [None])[0] if output.get("response_logprobs") else None

        # Use chat_parser to extract reasoning and tool_calls (matching VerlEngine)
        parsed_output = self.chat_parser.parse_completion(response_ids)
        content = parsed_output["content"]
        reasoning = parsed_output["reasoning"]
        tool_calls = parsed_output["tool_calls"]

        # Determine finish reason
        finish_reason = stop_reason
        if len(response_ids) >= sampling_params.get("max_tokens", self.max_response_length):
            finish_reason = "length"

        return ModelOutput(
            text=response_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=response_ids,
            prompt_length=prompt_length,
            completion_length=len(response_ids),
            finish_reason=finish_reason,
            logprobs=logprobs,
        )

    async def wake_up(self, tags=None):
        """Wake up the inference engine (for colocated training).
        
        Args:
            tags: Optional list of tags for multi-stage wake up (e.g., ["weights"], ["kv_cache"])
        """
        if self.inference_engine is not None:
            await self.inference_engine.wake_up(tags=tags)

    async def sleep(self, tags=None):
        """Put the inference engine to sleep (for colocated training).
        
        Args:
            tags: Optional list of tags for multi-stage sleep
        """
        if self.inference_engine is not None:
            await self.inference_engine.sleep(tags=tags)

    async def shutdown(self):
        """Shutdown the inference engine (final cleanup).
        
        This should be called when the engine is no longer needed.
        """
        if self.inference_engine is not None:
            await self.inference_engine.teardown()

