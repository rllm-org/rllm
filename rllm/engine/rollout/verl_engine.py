import asyncio
import uuid
from typing import cast

from typing_extensions import override
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine, TokenInput, VerlTokenOutput
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
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

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    @override
    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> VerlTokenOutput:
        token_input = cast(list[int], token_input)

        input_length = len(token_input)
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        validate = self.validate or kwargs.pop("validate", False)

        if enforce_max_prompt_length and input_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        sampling_params = self.val_sampling_params.copy() if validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)
        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        # starting from verl 0.7.0, we can pass in per-turn max_tokens into the sampling_params
        sampling_params["max_tokens"] = max_tokens

        token_output = await self.server_manager.generate(request_id=application_id, prompt_ids=token_input, sampling_params=sampling_params)
        return token_output

    @override
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
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

        prompt_length = len(prompt_ids)

        token_output: TokenOutput = await self.get_token_output_from_token_input(token_input=request_prompt_ids, **kwargs)
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

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])
