from typing import Any

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class MultiTurnWorkflow(TimingTrackingMixin, Workflow):
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

    async def _generate_model_step(self, messages: list[dict], *, task: dict, application_id: str, **kwargs) -> tuple[ModelOutput, list[float] | None, dict | None]:  # noqa: ARG002
        output: ModelOutput = await self.timed_llm_call(messages, application_id=application_id, **kwargs)
        return output, None, None

    def _attach_model_step(
        self,
        current_step,
        output: ModelOutput,  # noqa: ARG002
        response_mask: list[float] | None,  # noqa: ARG002
        metadata: dict | None,  # noqa: ARG002
    ) -> None:
        return

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute a multi-step workflow"""

        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        for _ in range(1, self.max_steps + 1):
            output, response_mask, metadata = await self._generate_model_step(
                self.agent.chat_completions,
                application_id=uid,
                task=task,
                **kwargs,
            )
            response = output.text

            action = self.agent.update_from_model(response)
            current_step = self.agent.trajectory.steps[-1] if self.agent.trajectory.steps else None

            next_obs, reward, done, info = await self.timed_env_call(self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, info)
            self._attach_model_step(current_step, output, response_mask, metadata)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        super().reset(task, uid)
        return self.env.reset(task)
