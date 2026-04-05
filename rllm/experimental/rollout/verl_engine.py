import uuid
from typing import cast

from omegaconf import DictConfig
from typing_extensions import override

from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import TokenInput, Tokenizer, TokenOutput, VerlTokenOutput
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    def __init__(self, config: DictConfig, server_manager, tokenizer: Tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.server_manager = server_manager

        self.tokenizer = tokenizer
        self.processor = processor
        chat_template_kwargs = config.rllm.rollout.get("chat_template_kwargs", {})
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=chat_template_kwargs.get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = chat_template_kwargs.get("accumulate_reasoning", False)
        self.reasoning_effort = chat_template_kwargs.get("reasoning_effort", "medium")

        rllm_sampling = config.rllm.rollout.sampling
        self.train_sampling_params = dict(
            temperature=rllm_sampling.train.temperature,
            top_k=rllm_sampling.train.top_k,
            top_p=rllm_sampling.train.top_p,
            logprobs=1,
        )

        self.val_sampling_params = dict(
            temperature=rllm_sampling.val.temperature,
            top_k=rllm_sampling.val.top_k,
            top_p=rllm_sampling.val.top_p,
            logprobs=1,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    @override
    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> VerlTokenOutput:
        token_input = cast(list[int], token_input)

        input_length = len(token_input)
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        if enforce_max_prompt_length and input_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampling_params = self.val_sampling_params.copy() if self.is_validation else self.train_sampling_params.copy()
        sampling_params.update(kwargs)
        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        # starting from verl 0.7.0, we can pass in per-turn max_tokens into the sampling_params
        sampling_params["max_tokens"] = max_tokens

        token_output = await self.server_manager.generate(request_id=application_id, prompt_ids=token_input, sampling_params=sampling_params)
        return token_output

    @override
    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning, reasoning_effort=reasoning_effort)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            model_inputs = self.processor(text=[prompt], images=image_data)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        token_output: TokenOutput = await self.get_token_output_from_token_input(token_input=request_prompt_ids, **kwargs)
        extra_kwargs = dict(prompt_ids=prompt_ids, multi_modal_inputs=multi_modal_inputs)
        return self.assemble_model_output(token_input=request_prompt_ids, token_output=token_output, **extra_kwargs)

    @override
    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput, **kwargs) -> ModelOutput:
        prompt_ids = kwargs.pop("prompt_ids", None)
        multi_modal_inputs = kwargs.pop("multi_modal_inputs", None)
        prompt_length = len(prompt_ids) if prompt_ids is not None else 0

        token_output = cast(VerlTokenOutput, token_output)
        completion_ids = token_output.token_ids
        logprobs = token_output.log_probs

        # convert the stop reason from verl back to the standard finish reason TODO(listar2000): check backward-compatibility
        reason_mapping = {"aborted": "abort", "completed": "stop"}
        if token_output.stop_reason is not None:
            finish_reason = reason_mapping.get(token_output.stop_reason, token_output.stop_reason)
        else:
            finish_reason = "stop"

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        # Encode routed_experts to per-token base64 with shape header.
        # Input: np.ndarray (completion_len, num_layers, topk) int32
        # Output: list[str] where first entry is JSON shape header, rest are base64 per token
        routing_matrices = None
        if token_output.routed_experts is not None:
            import base64
            import json as _json
            arr = token_output.routed_experts  # (completion_len, num_layers, topk)
            shape_header = _json.dumps({"shape": list(arr.shape[1:])})
            routing_matrices = [shape_header] + [
                base64.b64encode(arr[i].tobytes()).decode("ascii")
                for i in range(len(arr))
            ]

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
            routing_matrices=routing_matrices,
            metrics={"num_preempted": token_output.num_preempted} if token_output.num_preempted else None,
        )
