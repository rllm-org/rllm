import asyncio
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
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
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

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
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        token_output: TokenOutput = await self.server_manager.generate(request_id=application_id, prompt_ids=request_prompt_ids, image_data=image_data, sampling_params=sampling_params)  # type: ignore
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

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
            logprobs=[],
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
