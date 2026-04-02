from typing import Any

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.early_finalize import attach_model_output_to_step, maybe_generate_with_early_finalize
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class SingleTurnWorkflow(TimingTrackingMixin, Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
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

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute a single-step workflow"""

        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        generation = await maybe_generate_with_early_finalize(
            self,
            self.agent.chat_completions,
            application_id=uid,
            task=task,
            skip_special_tokens=True,
            **kwargs,
        )
        output: ModelOutput = generation.output
        response = output.text

        action = self.agent.update_from_model(response)
        current_step = self.agent.trajectory.steps[-1] if self.agent.trajectory.steps else None
        attach_model_output_to_step(current_step, output, generation.response_mask)

        _, reward, done, info = await self.timed_env_call(self.env.step, action)
        self.agent.update_from_env({}, reward, done, info)
        if current_step is not None and generation.metadata is not None:
            current_step.info["early_finalize"] = generation.metadata

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        if done:
            raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        super().reset(task, uid)
        return self.env.reset(task)
