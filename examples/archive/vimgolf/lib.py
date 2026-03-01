"""VimGolf agents and environments implementation."""

import copy
import json
from typing import Any

import vimgolf_gym
import vimgolf_gym.dataclasses

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.environments import SingleTurnEnvironment
from rllm.rewards import RewardOutput


class VimGolfSingleTurnAgent(BaseAgent):
    """
    A single turn VimGolf Agent.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the VimGolfSingleTurnAgent.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # Format observation based on whether it's the initial problem or subsequent feedback
        if not self.trajectory.steps:
            # Initial problem presentation
            assert isinstance(observation, dict) and "question" in observation
            question = observation["question"]
            formatted_observation = question
        else:
            # Follow-up correction prompt (never used, to be changed in multi-turn agent)
            formatted_observation = "Your previous answer may contain a mistake. Please review it carefully and answer again."

        self.messages.append({"role": "user", "content": formatted_observation})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)

        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]


def vimgolf_reward_function(task_info: dict, action: str) -> RewardOutput:
    task_data_str = task_info.get("ground_truth")
    task_data = json.loads(task_data_str)

    input = task_data["input"]
    target = task_data["target"]
    challenge_id = task_data["id"]

    solution = get_last_non_empty_line(action)
    custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge(input=input, output=target, solution=solution, name=challenge_id)
    verified = run_vimgolf_local(custom_challenge)
    if verified:
        reward = 1.0
        is_correct = True
    else:
        reward = 0.0
        is_correct = False
    ret = RewardOutput(reward=reward, is_correct=is_correct, metadata={})
    return ret


def run_vimgolf_local(custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge):
    validated = False
    with vimgolf_gym.make(
        "vimgolf-custom",
        custom_challenge=custom_challenge,
    ) as env:
        if custom_challenge.solution:
            validated = env.verify_keys(custom_challenge.solution)
    return validated


def get_last_non_empty_line(content: str):
    lines = content.splitlines()
    lines = [it.strip() for it in lines if it.strip()]
    if lines:
        return lines[-1]
    else:
        return ""


class VimGolfSingleTurnEnv(SingleTurnEnvironment):
    """Single turn environment for VimGolf."""

    def __init__(self, task=None, reward_fn=None, **kwargs):
        super().__init__(task=task, reward_fn=vimgolf_reward_function, **kwargs)
