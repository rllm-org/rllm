"""Rollout engine for the Miles backend: token-in, token-out against Miles' SGLang router.

rLLM already owns tokenization, prompt assembly and loss masks end to end, so this
talks to SGLang's stateless ``/generate`` endpoint with ``input_ids`` rather than
going through Miles' TITO session server -- routing through the session server
would install a second tokenizer authority and duplicate the merge logic that
``rllm/trainer/miles/transform.py`` already performs.

Request/response shape mirrors ``miles/rollout/generate_utils/generate_endpoint_utils.py``
so the two stay interchangeable against the same router.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import cast

import httpx
import numpy as np
from omegaconf import DictConfig
from typing_extensions import override

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.engine.rollout.types import MilesTokenOutput, TokenInput, Tokenizer, TokenOutput
from rllm.parser import ChatTemplateParser
from rllm.types import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

# Agentic turns legitimately take minutes; the trainer bounds the episode, not this.
_REQUEST_TIMEOUT_S = 3600


class MilesEngine(RolloutEngine):
    def __init__(self, config: DictConfig, router_url: str, tokenizer: Tokenizer, miles_args=None, **kwargs):
        super().__init__()
        self.config = config
        self.router_url = router_url.rstrip("/")
        self.tokenizer = tokenizer
        self.miles_args = miles_args

        rllm_cfg = config.get("rllm", {})
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=rllm_cfg.get("disable_thinking", False))
        self.max_prompt_length = rllm_cfg.data.max_prompt_length
        self.max_response_length = rllm_cfg.data.max_response_length
        self.accumulate_reasoning = rllm_cfg.get("accumulate_reasoning", False)
        self.router_replay_mode = rllm_cfg.get("algorithm", {}).get("router_replay", "disabled")

        # Miles gates the extra return fields on its own args; mirror them so a run
        # that did not enable routing replay never asks the engine for it.
        self._return_routed_experts = bool(getattr(miles_args, "use_rollout_routing_replay", False))
        self._return_indexer_topk = bool(getattr(miles_args, "use_rollout_indexer_replay", False))

        # rllm.rollout.train / .val are the canonical sampling blocks (see
        # rllm/trainer/config/rllm/base.yaml); max_tokens is applied per request in
        # get_token_output_from_token_input, not here.
        rollout_cfg = rllm_cfg.get("rollout", {}) or {}
        self.train_sampling_params = self._sampling_from(rollout_cfg.get("train", {}))
        self.val_sampling_params = self._sampling_from(rollout_cfg.get("val", {})) or self.train_sampling_params.copy()

        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S)

    @staticmethod
    def _sampling_from(block) -> dict:
        """Pass through what SGLang accepts, dropping keys it does not."""
        if not block:
            return {}
        allowed = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty", "stop", "stop_token_ids")
        return {k: v for k in allowed if (v := block.get(k)) is not None}

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    async def close(self) -> None:
        await self._client.aclose()

    @override
    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> MilesTokenOutput:
        input_ids = cast(list[int], token_input)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        if enforce_max_prompt_length and len(input_ids) > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampling_params = (self.val_sampling_params if self.is_validation else self.train_sampling_params).copy()
        sampling_params.update(kwargs)
        max_new_tokens = sampling_params.pop("max_tokens", None) or sampling_params.pop("max_new_tokens", None) or self.max_response_length
        # The context window bounds the whole sequence, not just the completion.
        max_new_tokens = min(max_new_tokens, max(self.max_prompt_length + self.max_response_length - len(input_ids), 0))
        if max_new_tokens <= 0:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        payload = {
            "input_ids": input_ids,
            "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "return_routed_experts": self._return_routed_experts,
            "return_indexer_topk": self._return_indexer_topk,
        }

        response = await self._client.post(f"{self.router_url}/generate", json=payload)
        response.raise_for_status()
        output = response.json()

        meta = output.get("meta_info", {})
        # SGLang returns [logprob, token_id, ...] per position.
        pairs = meta.get("output_token_logprobs") or []
        token_ids = [item[1] for item in pairs]
        log_probs = [item[0] for item in pairs]

        finish_reason = meta.get("finish_reason")
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type")
        if finish_reason in ("abort", "aborted"):
            raise RuntimeError("Rollout aborted by the Miles router (weight update or oversampling abort)")

        return MilesTokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason="length" if len(token_ids) >= max_new_tokens else (finish_reason or "stop"),
            routed_experts=self._decode_routed_experts(meta),
        )

    @staticmethod
    def _decode_routed_experts(meta: dict):
        """R3: SGLang ships routed experts as a base64 blob plus a shape header."""
        blob = meta.get("output_routed_experts")
        if not blob:
            return None
        try:
            header, data = blob if isinstance(blob, list) else (None, blob)
            arr = np.frombuffer(base64.b64decode(data), dtype=np.int32)
            if header:
                shape = json.loads(header)["shape"]
                arr = arr.reshape(-1, *shape)
            return arr
        except Exception as e:  # a malformed blob must not kill the rollout
            logger.warning("Could not decode routed experts from the router response: %s", e)
            return None

    @override
    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", "medium")

        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
            reasoning_effort=reasoning_effort,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_output = await self.get_token_output_from_token_input(token_input=prompt_ids, **kwargs)
        return self.assemble_model_output(token_input=prompt_ids, token_output=token_output, prompt_ids=prompt_ids)

    @override
    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput, **kwargs) -> ModelOutput:
        prompt_ids = kwargs.pop("prompt_ids", None) or cast(list[int], token_input)
        token_output = cast(MilesTokenOutput, token_output)
        completion_ids = token_output.token_ids

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        parsed = self.chat_parser.parse_completion(completion_ids)

        routing_matrices = None
        if self.router_replay_mode == "R3" and token_output.routed_experts is not None:
            arr = token_output.routed_experts
            header = json.dumps({"shape": list(arr.shape[1:]), "dtype": str(arr.dtype)})
            routing_matrices = [header, base64.b64encode(arr.tobytes()).decode("ascii")]

        return ModelOutput(
            text=completion_text,
            content=parsed["content"],
            reasoning=parsed["reasoning"],
            tool_calls=parsed["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=token_output.log_probs,
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=token_output.stop_reason,
            routing_matrices=routing_matrices,
        )
