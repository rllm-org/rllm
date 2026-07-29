"""Fireworks serverless backend using the Tinker-compatible rLLM pipeline."""

from __future__ import annotations

import os
from typing import Any

from fireworks.training.sdk import FiretitanServiceClient
from omegaconf import DictConfig

from rllm.trainer.backend_protocol import BackendProtocol
from rllm.trainer.fireworks.fireworks_serverless_policy_trainer import (
    FireworksServerlessPolicyTrainer,
)
from rllm.trainer.tinker.tinker_backend import TinkerBackend


def serverless_base_url(base_url: str) -> str:
    """Normalize a Fireworks API URL to its serverless training endpoint."""
    root = base_url.rstrip("/")
    if root.endswith("/training/v1/serverless"):
        return root
    if root.endswith("/training/v1"):
        return f"{root}/serverless"
    return f"{root}/training/v1/serverless"


class FireworksServerlessBackend(TinkerBackend):
    """Train and sample from one pooled Fireworks serverless session."""

    name = "fireworks_serverless"

    def __init__(self, config: DictConfig, **kwargs: Any) -> None:
        # Deliberately skip TinkerBackend.__init__: it creates a Tinker
        # ServiceClient, while this backend needs Fireworks' pooled session.
        BackendProtocol.__init__(self, config, **kwargs)
        self.full_config = config
        self.service_client = FiretitanServiceClient(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url=serverless_base_url(config.fireworks_base_url),
        )
        self.policy_trainer = None
        self.tokenizer = None
        self.rollout_engine = None
        self.sampling_client = None
        self._algorithm_config = None
        self._policy_updated_this_step = False
        self.learning_rate = config.training.get("learning_rate", 1e-6)
        self.beta1 = config.training.get("beta1", 0.9)
        self.beta2 = config.training.get("beta2", 0.95)
        self.eps = config.training.get("eps", 1e-8)

    def init_rollout_engine(self, **kwargs: Any):
        rollout_engine = super().init_rollout_engine(**kwargs)
        self.policy_trainer = FireworksServerlessPolicyTrainer(
            config=self.full_config,
            service_client=self.service_client,
            tokenizer=self.tokenizer,
            cf_config=kwargs.get("cf_config"),
            transform_config=kwargs.get("transform_config"),
            algorithm_config=kwargs.get("algorithm_config"),
        )
        return rollout_engine

    def validate_config(self) -> None:
        super().validate_config()
        if not os.environ.get("FIREWORKS_API_KEY"):
            raise ValueError("FIREWORKS_API_KEY is required for Fireworks serverless training")
        if self.full_config.model.lora_rank <= 0:
            raise ValueError("Fireworks serverless training requires model.lora_rank > 0")
        if not self.full_config.training.max_length:
            raise ValueError(
                "Fireworks serverless training requires training.max_length "
                "(there is no training shape to infer it from)"
            )

    async def on_policy_updated(self, trainer_state) -> None:
        previous_client = self.sampling_client
        await super().on_policy_updated(trainer_state)
        if previous_client is not None and previous_client is not self.sampling_client:
            previous_client.close()

    def shutdown(self) -> None:
        try:
            if self.sampling_client is not None:
                self.sampling_client.close()
                self.sampling_client = None
        finally:
            self.service_client.close()
            super().shutdown()
