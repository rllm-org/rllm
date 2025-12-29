"""
SkyRL rollout engine implementation.

This is a wrapper around SkyRL's InferenceEngineClient to provide
the RolloutEngine interface for rLLM workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.trainer import RayPPOTrainer
    from transformers import PreTrainedTokenizer
    from omegaconf import DictConfig


class SkyRLEngine(RolloutEngine):
    """
    RolloutEngine implementation using SkyRL's InferenceEngineClient for model inference.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        inference_engine_client: InferenceEngineClient | None = None,
        **kwargs,
    ):
        """
        Initialize SkyRLEngine.

        Args:
            config: Configuration object
            tokenizer: Tokenizer for encoding/decoding
            inference_engine_client: SkyRL InferenceEngineClient (optional, can be set later)
            **kwargs: Additional arguments
        """
        self.config = config
        self.tokenizer = tokenizer
        self.inference_engine_client = inference_engine_client
        self.skyrl_trainer: RayPPOTrainer | None = None

    def set_skyrl_components(
        self,
        inference_engine_client: InferenceEngineClient | None = None,
        trainer: RayPPOTrainer | None = None,
    ):
        """Set SkyRL components after initialization.

        Args:
            inference_engine_client: SkyRL InferenceEngineClient
            trainer: SkyRL RayPPOTrainer
        """
        if inference_engine_client is not None:
            self.inference_engine_client = inference_engine_client
        if trainer is not None:
            self.skyrl_trainer = trainer

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """
        Generate model response for a given set of messages.

        Args:
            messages: List of message dictionaries (OpenAI format)
            **kwargs: Additional parameters

        Returns:
            ModelOutput with generated text and metadata
        """
        if self.inference_engine_client is None:
            raise RuntimeError("InferenceEngineClient not set. Call set_skyrl_components() first.")

        # Convert messages to SkyRL format and call inference engine
        # This is a simplified implementation - you may need to adjust based on
        # your specific SkyRL setup and how it handles message formatting

        # For now, we'll use a placeholder that needs to be implemented
        # based on your SkyRL InferenceEngineClient API
        raise NotImplementedError(
            "SkyRLEngine.get_model_response() needs to be implemented based on "
            "your SkyRL InferenceEngineClient API. You may need to convert "
            "messages to the format expected by SkyRL and call the appropriate "
            "inference methods."
        )

