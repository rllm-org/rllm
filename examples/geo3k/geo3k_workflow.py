import base64
from io import BytesIO

from PIL import Image

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction, math_reward_fn
from rllm.workflows.simple_workflow import SimpleAgent
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class Geo3KWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction = None, encode_as_base64: bool = False, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.agent = SimpleAgent()
        self.reward_fn: RewardFunction = reward_function or math_reward_fn
        self.encode_as_base64 = encode_as_base64

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        question = task.get("question")
        image = task.get("image", task.get("images", None))
        if isinstance(image, list) and len(image) > 0:
            image = image[0]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        assert isinstance(image, Image.Image) or image is None, f"Image must be a PIL.Image.Image, but got {type(image)}"

        if self.encode_as_base64 and image is not None:
            # format as openai compatible base64 encoded image
            image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ]
        elif image is not None:
            messages = [{"role": "user", "content": question, "images": [image]}]
        else:
            messages = [{"role": "user", "content": question}]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)
        action = Action(output.content)
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

        self.commit(agent=self.agent, reset=True)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)
