"""RolloutEngine backed by Fireworks ``FiretitanSamplingClient``.

FiretitanSamplingClient is compatible with the Tinker sampling client surface,
so token-in / token-out sampling is inherited from ``TinkerEngine``. Fireworks
only customizes client setup and Fireworks-specific sampling parameters.
"""

from __future__ import annotations

from fireworks.training.sdk import FiretitanSamplingClient
from typing_extensions import override

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.engine.rollout.types import Tokenizer



class FireworksEngine(TinkerEngine):
    """``TinkerEngine`` subclass that uses a Fireworks ``FiretitanSamplingClient``
    for inference instead of a Tinker ``SamplingClient``.

    ``FiretitanSamplingClient`` supports token-in / token-out via the
    ``/inference/v1/completions`` endpoint, so ``TinkerTokenInput`` and
    ``TinkerTokenOutput`` are fully supported.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sampling_client: FiretitanSamplingClient,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        max_model_length: int = 32768,
        sampling_params: dict | None = None,
        disable_thinking: bool = False,
        accumulate_reasoning: bool = False,
        reasoning_effort: str = "medium",
        sample_timeout: int = 600,
        processor=None,
        router_replay: bool = False,
        **kwargs,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer for chat-template rendering.
            sampling_client: Pre-built ``FiretitanSamplingClient``.
            max_prompt_length: Hard cap on prompt token length.
            max_response_length: Default max completion tokens.
            max_model_length: Total context window.
            sampling_params: Dict with optional ``"train"`` / ``"val"``
                sub-dicts for default sampling kwargs.
            disable_thinking: Suppress thinking tokens in the prompt.
            accumulate_reasoning: Accumulate reasoning across turns.
            reasoning_effort: Reasoning effort for the chat parser and Fireworks
                completions API (e.g. ``low``, ``medium``, ``high``, ``none``).
            sample_timeout: HTTP timeout (seconds) for sampling calls.
            processor: Optional ``ProcessorMixin`` for multimodal models.
            router_replay: If True, request and propagate routing matrices
                for Router Replay (R3) training.
        """
        from rllm.engine.rollout.rollout_engine import RolloutEngine
        from rllm.parser import ChatTemplateParser

        # Skip TinkerEngine.__init__ (it requires tinker.ServiceClient);
        # set up the same attributes directly.
        RolloutEngine.__init__(self)

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.accumulate_reasoning = accumulate_reasoning
        self.reasoning_effort = reasoning_effort

        self.train_sampling_params = dict((sampling_params or {}).get("train", {}))
        self.val_sampling_params = dict((sampling_params or {}).get("val", {}))
        if reasoning_effort is not None:
            self.train_sampling_params.setdefault("reasoning_effort", reasoning_effort)
            self.val_sampling_params.setdefault("reasoning_effort", reasoning_effort)
        if router_replay:
            self.train_sampling_params["include_routing_matrix"] = True
            self.val_sampling_params["include_routing_matrix"] = True

        # Chat template parser (same setup as TinkerEngine bypass mode)
        self.bypass_render_with_parser = True
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=disable_thinking,
        )
        if hasattr(self.chat_parser, "stop_sequences") and self.chat_parser.stop_sequences:
            self.stop_sequences = self.chat_parser.stop_sequences
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
            self.stop_sequences = [tokenizer.eos_token_id]
        else:
            raise ValueError("No stop sequences found for tokenizer or chat parser")

        self.sample_timeout = sample_timeout
        self.router_replay = router_replay
        self.sampling_client = sampling_client

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", None)

        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            reasoning_effort=reasoning_effort,
            accumulate_reasoning=accumulate_reasoning,
        )
        token_input = self.tokenizer.encode(prompt, add_special_tokens=False)

        if application_id is not None:
            kwargs["user"] = application_id

        version = self.weight_version
        sampled_sequence = await self.get_token_output_from_token_input(
            token_input=token_input,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        result = self.assemble_model_output(token_input=token_input, token_output=sampled_sequence)
        result.weight_version = version
        result.routing_matrices = getattr(sampled_sequence, "routing_matrices", None)
        result.metrics = getattr(sampled_sequence, "server_metrics", None)
        return result

    @override
    async def get_model_response_from_tokens(self, token_input, **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", None)
        if application_id is not None:
            kwargs["user"] = application_id

        version = self.weight_version
        sampled_sequence = await self.get_token_output_from_token_input(token_input=token_input, **kwargs)
        result = self.assemble_model_output(token_input=token_input, token_output=sampled_sequence)
        result.weight_version = version
        result.routing_matrices = getattr(sampled_sequence, "routing_matrices", None)
        result.metrics = getattr(sampled_sequence, "server_metrics", None)
        return result

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    async def compute_logprobs(self, ids: list[int]) -> list[float]:
        raise NotImplementedError("compute_logprobs is not supported by FireworksEngine.")