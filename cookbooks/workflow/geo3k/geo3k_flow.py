"""Geo3K Workflow — VLM geometry problem solver (legacy Workflow API).

A single-turn VLM agent that solves geometry problems from the Geometry3K
dataset. Built on the ``rllm.workflows.workflow.Workflow`` base class +
``RolloutEngine`` (the pre-AgentFlow API kept around for backward
compatibility).

For the equivalent example written against the newer ``@rllm.rollout``
AgentFlow protocol, see ``cookbooks/geo3k/``.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction, math_reward_fn
from rllm.types import Action, Episode, Step, Trajectory
from rllm.workflows.simple_workflow import SimpleAgent
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class Geo3KWorkflow(Workflow):
    """Single-turn VLM geometry solver.

    The workflow assembles a ``{"role": "user", "content": <text>,
    "images": [PIL.Image]}`` message and calls
    ``rollout_engine.get_model_response`` directly. The backend-specific
    image rendering (OpenAI multimodal blocks for verl/vLLM, the
    tinker-renderer-aware path for tinker) happens inside the
    ``RolloutEngine`` itself.
    """

    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.agent = SimpleAgent()
        self.reward_fn: RewardFunction = reward_function or math_reward_fn

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        question = task.get("question")
        image = task.get("image", task.get("images", None))
        if isinstance(image, list) and len(image) > 0:
            image = image[0]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        assert isinstance(image, Image.Image) or image is None, f"Image must be a PIL.Image.Image, but got {type(image)}"

        # Standard format: content is text, images is list[PIL.Image].
        # Conversion to backend-specific format happens in the rollout engine.
        if image is not None:
            messages = [{"role": "user", "content": question, "images": [image]}]
        else:
            messages = [{"role": "user", "content": question}]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)
        action = Action(action=output.content)
        reward_result = self.reward_fn(task, action)

        trajectory: Trajectory = self.agent.trajectory
        trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                thought=output.reasoning,
                action=action,
                reward=reward_result.reward,
                model_output=output,
            )
        )

        self.commit(name="solver", agent=self.agent, reset=True)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)
