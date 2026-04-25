from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.environments.base.base_env import BaseEnv
from rllm.workflows.early_finalize import EarlyFinalizeResult
from rllm.workflows.early_finalize_workflows import EarlyFinalizeWorkflowMixin, MultiTurnWorkflowWithEarlyFinalize
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason


class DummyAgent(BaseAgent):
    def __init__(self):
        self._trajectory = Trajectory()
        self._messages: list[dict[str, str]] = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return list(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation, reward: float, done: bool, info: dict, **kwargs):  # noqa: ARG002
        if observation is None or observation == {}:
            if self._trajectory.steps:
                cur_step = self._trajectory.steps[-1]
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        text = observation["question"] if isinstance(observation, dict) else str(observation)
        self._messages.append({"role": "user", "content": text})
        self._trajectory.steps.append(Step(observation=text))

    def update_from_model(self, response: str, **kwargs) -> Action:  # noqa: ARG002
        self._messages.append({"role": "assistant", "content": response})
        cur_step = self._trajectory.steps[-1]
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response
        cur_step.action = Action(action=response)
        return cur_step.action

    def reset(self):
        self._trajectory = Trajectory()
        self._messages = []


class DummyEnv(BaseEnv):
    def reset(self, task: dict) -> tuple[dict, dict]:
        return {"question": task["question"]}, {}

    def step(self, action) -> tuple[dict, float, bool, dict]:  # noqa: ARG002
        return {}, 1.0, True, {}

    @staticmethod
    def from_dict(info: dict) -> DummyEnv:  # noqa: ARG004
        return DummyEnv()


class DummyRolloutEngine:
    def __init__(self, output: ModelOutput | None = None):
        self.output = output or ModelOutput(
            text="answer",
            content="answer",
            prompt_ids=[1],
            completion_ids=[2],
            logprobs=[-0.1],
            prompt_length=1,
            completion_length=1,
            finish_reason="stop",
        )
        self.calls = 0
        self.max_response_length = 256

    async def get_model_response(self, messages, **kwargs):  # noqa: ARG002
        self.calls += 1
        return self.output


class PromptGuardRolloutEngine(DummyRolloutEngine):
    def __init__(self):
        super().__init__()
        self.max_prompt_length = 2000
        self.max_response_length = 500
        self.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(range(1601)))  # noqa: ARG005
        self.chat_parser = SimpleNamespace(parse=lambda *args, **kwargs: "prompt")


class PromptGuardWorkflow(MultiTurnWorkflow):
    async def run(self, task: dict, uid: str, **kwargs):
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0, False, info)

        max_model_len = self.rollout_engine.max_prompt_length + self.rollout_engine.max_response_length
        min_response_buffer = 1000
        prompt = self.rollout_engine.chat_parser.parse(
            self.agent.chat_completions,
            add_generation_prompt=True,
            is_first_msg=True,
        )
        prompt_length = len(self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False))

        if prompt_length > max_model_len - min_response_buffer:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        await self._generate_model_step(
            self.agent.chat_completions,
            application_id=uid,
            task=task,
            enforce_max_prompt_length=False,
            **kwargs,
        )
        raise AssertionError("prompt guard should have terminated before generation")


class PromptGuardWorkflowWithEarlyFinalize(EarlyFinalizeWorkflowMixin, PromptGuardWorkflow):
    pass


def test_default_multi_turn_workflow_keeps_plain_generation_behavior():
    rollout_engine = DummyRolloutEngine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = MultiTurnWorkflow(
            agent_cls=DummyAgent,
            env_cls=DummyEnv,
            max_steps=1,
            rollout_engine=rollout_engine,
            executor=executor,
        )
        episode = asyncio.run(workflow.run_with_termination_handling({"question": "question"}, "task:0"))

    step = episode.trajectories[0].steps[0]
    assert episode.termination_reason == TerminationReason.ENV_DONE
    assert rollout_engine.calls == 1
    assert step.model_output is None
    assert "early_finalize" not in step.info


def test_explicit_multi_turn_workflow_uses_early_finalize_helper(monkeypatch):
    output = ModelOutput(
        text="rescued answer",
        content="rescued answer",
        prompt_ids=[1],
        completion_ids=[2],
        logprobs=[-0.1],
        prompt_length=1,
        completion_length=1,
        finish_reason="stop",
    )
    called = {"value": False}

    async def fake_maybe_generate(workflow, messages, *, application_id: str, task=None, **kwargs):  # noqa: ARG001
        called["value"] = True
        assert workflow.early_finalize_config.enable is True
        return EarlyFinalizeResult(
            output=output,
            response_mask=[1.0],
            metadata={"attempted": True},
        )

    monkeypatch.setattr("rllm.workflows.early_finalize_workflows.maybe_generate_with_early_finalize", fake_maybe_generate)

    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = MultiTurnWorkflowWithEarlyFinalize(
            agent_cls=DummyAgent,
            env_cls=DummyEnv,
            max_steps=1,
            rollout_engine=DummyRolloutEngine(output=output),
            executor=executor,
            early_finalize_config={"reserve_response_tokens": 8},
        )
        episode = asyncio.run(workflow.run_with_termination_handling({"question": "question"}, "task:0"))

    step = episode.trajectories[0].steps[0]
    assert called["value"] is True
    assert episode.termination_reason == TerminationReason.ENV_DONE
    assert step.model_output is not None
    assert step.model_output.completion_ids == [2]
    assert step.response_mask == [1.0]
    assert step.info["early_finalize"]["attempted"] is True


def test_opt_in_workflow_keeps_prompt_guard_ahead_of_early_finalize(monkeypatch):
    async def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("early finalize helper should not run when the prompt guard triggers first")

    monkeypatch.setattr("rllm.workflows.early_finalize_workflows.maybe_generate_with_early_finalize", fail_if_called)

    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow = PromptGuardWorkflowWithEarlyFinalize(
            agent_cls=DummyAgent,
            env_cls=DummyEnv,
            max_steps=1,
            rollout_engine=PromptGuardRolloutEngine(),
            executor=executor,
            early_finalize_config={"reserve_response_tokens": 8},
        )
        episode = asyncio.run(workflow.run_with_termination_handling({"question": "question"}, "task:0"))

    assert episode.termination_reason == TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
