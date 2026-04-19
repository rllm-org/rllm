import uuid
from typing import cast

from omegaconf import DictConfig
from typing_extensions import override
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager

from rllm.experimental.parser import get_chat_parser
from rllm.experimental.rollout.rollout_engine import CHAT_TEMPLATE_KWARG_NAMES, ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import TokenInput, Tokenizer, TokenOutput, VerlTokenOutput
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    def __init__(self, config: DictConfig, rollout_manager: AgentLoopManager, tokenizer: Tokenizer, processor=None, **kwargs):
        super().__init__()
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        assert rollout_manager.global_load_balancer is not None, "global_load_balancer is not available. Issues with RayPPOTrainer's `init_workers()` function."

        self.rollout_manager: AgentLoopManager = rollout_manager
        # reconstruct the servers list from the server_addresses and server_handles (Verl 0.7.0+)
        servers = zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True)
        self.server_manager = AsyncLLMServerManager(config, servers=servers, load_balancer_handle=rollout_manager.global_load_balancer)

        self.tokenizer = tokenizer
        self.processor = processor

        rollout_cfg = config.get("rollout", {}) or {}
        ctk = dict(rollout_cfg.get("chat_template_kwargs", {}) or {})
        parser_backend = ctk.pop("parser_backend", "rllm")
        reasoning_parser_name = ctk.pop("reasoning_parser_name", None)
        tool_parser_name = ctk.pop("tool_parser_name", None)
        renderer_name = ctk.pop("renderer_name", None)
        disable_thinking = ctk.pop("disable_thinking", False)
        chat_template = ctk.pop("chat_template", None)
        if chat_template:
            from pathlib import Path

            tokenizer.chat_template = Path(chat_template).read_text()
        self.default_template_kwargs: dict = dict(tools=[], **ctk)

        self.chat_parser = get_chat_parser(
            tokenizer,
            processor=processor,
            parser_backend=parser_backend,
            reasoning_parser_name=reasoning_parser_name,
            tool_parser_name=tool_parser_name,
            renderer_name=renderer_name,
            disable_thinking=disable_thinking,
        )

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length

        sp = rollout_cfg.get("sampling", {}) or {}
        self.train_sampling_params = dict(sp.get("train", {}) or {})
        self.val_sampling_params = dict(sp.get("val", {}) or {})
        self.train_sampling_params.setdefault("logprobs", 1)
        self.val_sampling_params.setdefault("logprobs", 1)

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
        template_kwargs = self.default_template_kwargs.copy()
        template_kwargs.update(kwargs.pop("chat_template_kwargs", {}))
        template_kwargs.update({k: kwargs.pop(k) for k in list(kwargs) if k in CHAT_TEMPLATE_KWARG_NAMES})

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, **template_kwargs)
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
        return self.assemble_model_output(token_input=prompt_ids, token_output=token_output, multi_modal_inputs=multi_modal_inputs)

    @override
    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput, **kwargs) -> ModelOutput:
        prompt_ids = list(token_input) if token_input is not None else None
        multi_modal_inputs = kwargs.pop("multi_modal_inputs", None)
        prompt_length = len(prompt_ids) if prompt_ids is not None else 0

        token_output = cast(VerlTokenOutput, token_output)
        completion_ids = token_output.token_ids
        logprobs = token_output.log_probs

        # convert the stop reason from verl back to the standard finish reason TODO(listar2000): check backward-compatibility
        reason_mapping = {"aborted": "error", "abort": "error", "completed": "stop"}
        if token_output.stop_reason is not None:
            finish_reason = reason_mapping.get(token_output.stop_reason, token_output.stop_reason)
        else:
            finish_reason = "stop"

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
