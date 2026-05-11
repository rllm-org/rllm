import logging
import uuid
from typing import cast

import torch
from omegaconf import DictConfig
from typing_extensions import override
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import Processor, TokenInput, Tokenizer, TokenOutput, VerlTokenOutput
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


class VerlEngine(RolloutEngine):
    def __init__(self, config: DictConfig, server_manager: AsyncLLMServerManager, tokenizer: Tokenizer, processor: Processor | None = None, **kwargs):
        super().__init__()
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.server_manager = server_manager

        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
            logprobs=1,
        )

        logger.info(f"train_sampling_params: {self.train_sampling_params}")
        logger.info(f"val_sampling_params: {self.val_sampling_params}")

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    @override
    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> VerlTokenOutput:
        token_input = cast(list[int], token_input)

        input_length = len(token_input)
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        # Multimodal: verl's AsyncLLMServerManager.generate accepts image_data /
        # video_data (list of PIL.Image / video tensors) and the underlying
        # vLLM server expands the per-image <|image_pad|> placeholder in
        # ``prompt_ids`` based on each image's actual grid size.
        image_data = kwargs.pop("image_data", None)
        video_data = kwargs.pop("video_data", None)

        if enforce_max_prompt_length and input_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampling_params = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()
        sampling_params.update(kwargs)
        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        # starting from verl 0.7.0, we can pass in per-turn max_tokens into the sampling_params
        sampling_params["max_tokens"] = max_tokens

        token_output = await self.server_manager.generate(
            request_id=application_id,
            prompt_ids=token_input,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )

        if token_output.stop_reason in ("aborted", "abort"):
            raise RuntimeError("Rollout aborted")
        token_output.stop_reason = "length" if len(token_output.token_ids) >= max_tokens else "stop"

        return token_output

    @override
    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", "medium")

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning, reasoning_effort=reasoning_effort)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            assert hasattr(self.chat_parser, "process_image_data"), f"Chat parser cls {self.chat_parser.__class__.__name__} does not have process_image_data method"
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            # Below we mirrors verl's ``agent_loop._compute_multi_modal_inputs``
            model_inputs = self.processor(text=[prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)

            grid_thw = multi_modal_inputs.get("image_grid_thw")
            if grid_thw is not None:
                multi_modal_inputs["images_seqlens"] = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        token_output: TokenOutput = await self.get_token_output_from_token_input(token_input=request_prompt_ids, image_data=image_data, **kwargs)
        extra_kwargs = dict(prompt_ids=prompt_ids, multi_modal_inputs=multi_modal_inputs)
        return self.assemble_model_output(token_input=request_prompt_ids, token_output=token_output, **extra_kwargs)

    @override
    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput, **kwargs) -> ModelOutput:
        # TITOCompleter calls this with just (token_input, token_output) — no kwargs.
        # In that path, ``token_input`` IS the prompt (the completer manages the
        # buffer manually), so default ``prompt_ids = token_input`` so downstream
        # transform logic (which reads ``model_output.prompt_ids``) still works.
        # The non-TITO ``_get_model_response`` path explicitly passes
        # ``prompt_ids=...`` to override this default with the "logical" prompt
        # (i.e. without any engine-internal prepends).
        prompt_ids = kwargs.pop("prompt_ids", None)
        if prompt_ids is None:
            prompt_ids = list(token_input)
        multi_modal_inputs = kwargs.pop("multi_modal_inputs", None)
        prompt_length = len(prompt_ids)

        token_output = cast(VerlTokenOutput, token_output)
        completion_ids = token_output.token_ids
        logprobs = token_output.log_probs
        finish_reason = token_output.stop_reason

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )
