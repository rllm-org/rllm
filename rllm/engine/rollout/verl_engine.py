import uuid

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser, ToolParser
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, **kwargs):
        self.config = config
        self.rollout_manager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, rollout_manager.async_llm_servers)
        self.tokenizer = tokenizer
        self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))

        try:
            self.tool_parser = ToolParser.get_parser(self.tokenizer)
        except Exception:
            print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
            self.tool_parser = None

        self.validate = False

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)

        if validate:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                top_k=self.config.actor_rollout_ref.rollout.val_kwargs.top_k,
                top_p=self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
            )
        else:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.do_sample is False else self.config.actor_rollout_ref.rollout.temperature,
                top_k=self.config.actor_rollout_ref.rollout.top_k,
                top_p=self.config.actor_rollout_ref.rollout.top_p,
            )
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", self.config.data.max_response_length)

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response_ids = await self.server_manager.generate(request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params)

        # verl sets max_tokens as max_model_len - len(prompt_ids), where max_model_len is config.data.max_prompt_length + config.data.max_response_length
        # so we truncate the response to max_tokens if it exceeds max_tokens
        finish_reason = "stop"
        if len(response_ids) >= max_tokens:
            finish_reason = "length"
            response_ids = response_ids[:max_tokens]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        tool_calls = None
        if self.tool_parser is not None:
            tool_calls = self.tool_parser.parse(response_text)

        return ModelOutput(text=response_text, tool_calls=tool_calls, finish_reason=finish_reason, completion_tokens=len(response_ids), prompt_tokens=len(prompt_ids))

    def wake_up(self):
        self.rollout_manager.wake_up()

    def sleep(self):
        self.rollout_manager.sleep()
