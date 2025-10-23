from typing import Any

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class CumulativeWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        max_steps=5,
        **kwargs,
    ):
        from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING

        super().__init__(**kwargs)

        agent_cls = AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
        env_cls = ENV_CLASS_MAPPING[env_cls] if isinstance(env_cls, str) else env_cls

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}

        self.agent = agent_cls(**agent_args)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps

        self.prompt_length = 0

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute a multi-step workflow"""

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        for i in range(1, self.max_steps + 1):
            prompt = self.rollout_engine.chat_parser.parse(self.agent.chat_completions, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=True)
            prompt_length = len(self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False))
            if i == 1:
                self.prompt_length = prompt_length
            max_tokens = self.rollout_engine.max_response_length - (prompt_length - self.prompt_length)
            if max_tokens <= 0:
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            output: ModelOutput = await self.rollout_engine.get_model_response(self.agent.chat_completions, application_id=uid, accumulate_reasoning=True, enforce_max_prompt_length=False, max_tokens=max_tokens, **kwargs)
            response = output.text

            action = self.agent.update_from_model(response)

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        super().reset(task, uid)
        self.prompt_length = 0
        return self.env.reset(task)
