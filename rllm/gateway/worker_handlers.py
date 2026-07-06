"""Handler factories for running the model gateway in a separate process.

The gateway's in-process ``local_handler`` (a rollout engine) can't cross a
process boundary, so a separate-process gateway rebuilds an equivalent one from
a serializable spec. ``GatewayManager`` obtains the spec via
``engine.handler_factory_spec()`` -> ``(import_path, config)`` and spawns
``python -m rllm_model_gateway --handler-factory <import_path>
--handler-config <config.json>``; the gateway process then imports the factory
named here and calls it with the config to build its ``local_handler``.

Each backend supplies its own factory (Fireworks below; Tinker would add one
that recreates a ``SamplingClient`` from a ``sampler_path``). The generic
multi-process machinery in the gateway package stays backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


def build_fireworks_handler(config: dict[str, Any]) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Rebuild a Fireworks ``local_handler`` from ``FireworksEngine.handler_factory_spec``.

    Attaches to the *same* deployment as the trainer (via ``inference_url`` +
    ``model``) — this is just a fresh HTTP client, not a re-provision. The API
    key is read from ``FIREWORKS_API_KEY`` (inherited env), never serialized.
    """
    from fireworks.training.sdk import DeploymentSampler
    from transformers import AutoTokenizer

    from rllm.engine.rollout.fireworks_engine import FireworksEngine
    from rllm.gateway.tinker_adapter import create_tinker_handler

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set in the gateway process env; cannot build the Fireworks sampler.")

    tokenizer_model = config["tokenizer_model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    sampler = DeploymentSampler(
        inference_url=config["inference_url"],
        model=config["model"],
        api_key=api_key,
        tokenizer=tokenizer,
    )

    engine = FireworksEngine(
        tokenizer=tokenizer,
        sampler=sampler,
        max_prompt_length=config["max_prompt_length"],
        max_response_length=config["max_response_length"],
        max_model_length=config["max_model_length"],
        sampling_params=config.get("sampling_params"),
        reasoning_effort=config.get("reasoning_effort", "medium"),
        accumulate_reasoning=config.get("accumulate_reasoning", False),
        router_replay=config.get("router_replay", False),
        sample_timeout=config.get("sample_timeout", 600),
        renderer_family=config.get("renderer_family", "auto"),
        bypass_render_with_parser=config.get("bypass_render_with_parser", False),
    )
    logger.info(
        "Built Fireworks gateway handler (deployment=%s, tokenizer=%s, renderer_family=%s)",
        config.get("model"),
        tokenizer_model,
        config.get("renderer_family"),
    )
    return create_tinker_handler(engine)
