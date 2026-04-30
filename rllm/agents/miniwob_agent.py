# DISCLAIMER:
# This MiniWoB agent implementation is adapted from the BrowserGym demo agent:
# https://github.com/ServiceNow/BrowserGym/blob/main/demo_agent/agent.py
# Some parts have been modified or extended for custom use.

import base64
import copy
import io
import logging
import re
from dataclasses import asdict
from typing import Any

import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet  # type: ignore[import-untyped]
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html  # type: ignore[import-untyped]
from PIL import Image

from rllm.agents.agent import BaseAgent
from rllm.agents.system_prompts import *
from rllm.types import Action, Step, Trajectory

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image) -> str:
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class MiniWobAgent(BaseAgent):
    def __init__(
        self,
        chat_mode: bool = False,
        use_html: bool = True,
        use_axtree: bool = True,
        use_screenshot: bool = False,
        use_accumulate_thinking: bool = True,
        cot_prompt: bool = False,
        use_full_conversation: bool = True,
        use_reward_shaping: bool = False,
    ):
        self.chat_mode: bool = chat_mode
        self.use_html: bool = use_html
        self.use_axtree: bool = use_axtree
        self.use_screenshot: bool = use_screenshot

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode="off",  # add visual effects
        )

        self.action_history: list[str] = []  # all are in string

        # for interface compliance
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.step: int = 0
        self.current_observation: dict[str, Any] | None = None
        self.reset()

        self.accumulate_thinking: bool = use_accumulate_thinking
        self.cot_prompt: bool = cot_prompt
        self.full_conversation: bool = use_full_conversation
        self.reward_shaping: bool = use_reward_shaping

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs) -> None:
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        obs = self._preproc_obs(observation)
        # Base message for the user
        user_prompt_content = self._format_msgs_as_str(self.get_user_msgs(obs))

        # initial state
        if not self.messages:
            self.messages.append(
                {"role": "system", "content": self._format_msgs_as_str(self.get_system_msgs(obs))},
            )

        self.messages.append({"role": "user", "content": user_prompt_content})
        self.current_observation = obs

        if done and self.reward_shaping:
            reward_penalty = 0.0
            for step in self.trajectory.steps:
                if not self.validate_step(asdict(step)):
                    reward_penalty = -0.5
                    break
            self.trajectory.reward += reward_penalty

    def update_from_model(self, response: str, **kwargs) -> Action:
        action_str = self._parse_model_response(response)

        self.messages.append({"role": "assistant", "content": response})
        self.step += 1

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), action=action_str, model_response=response, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        action_history_str = action_str if action_str != response else "Response is missing ``` ```"
        self.action_history.append(action_history_str)

        return Action(action=action_str)

    def _remove_thinking(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for msg in messages[:-1]:
            if msg["role"] == "assistant":
                _, sep, after = msg["content"].partition("</think>")
                if sep:
                    msg["content"] = after
        return messages

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            messages = self._remove_thinking(messages)

        if self.full_conversation:
            return messages

        latest_msgs = [self.messages[0]]  # system message
        has_assistant_msg = False
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                latest_msgs += self.messages[i + 1 :]
                has_assistant_msg = True
                break
        if not has_assistant_msg:
            latest_msgs += self.messages[1:]
        return latest_msgs

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.action_history = []
        self.step = 0

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            raise ValueError("get_current_state called before the first observation was processed.")
        return self._trajectory.steps[-1]

    def get_system_msgs(self, obs) -> list[dict[str, str]]:
        system_msgs = []
        system_msgs.append({"type": "text", "text": self._get_system_prompt()})

        # Add goal information
        system_msgs.append({"type": "text", "text": "\n# Goal (Below is the goal you want to accomplish):\n\n"})
        system_msgs.extend(obs["goal_object"])
        return system_msgs

    def get_user_msgs(self, user_obs) -> list[dict[str, str]]:
        user_msgs = []
        # Add open tabs information
        user_msgs.extend(self._format_open_tabs(user_obs["open_pages_urls"], user_obs["open_pages_titles"], user_obs["active_page_index"]))

        # Add page information based on settings
        if self.use_axtree:
            user_msgs.append({"type": "text", "text": f"# Current page Accessibility Tree\n\n{user_obs['axtree_txt']}\n\n"})

        if self.use_html:
            user_msgs.append({"type": "text", "text": f"# Current page DOM\n\n{user_obs['pruned_html']}\n\n"})

        if self.use_screenshot:
            user_msgs.extend(self._format_screenshot(user_obs["screenshot"]))

        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
# History of past actions
""",
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\
Action {i}:
{action}
"""
                        if i != len(self.action_history) - 1
                        else f"""\
Last Action:
{action}
""",
                    }
                    for i, action in enumerate(self.action_history)
                ]
            )

        if user_obs["last_action_error"]:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action

{user_obs["last_action_error"]}

""",
                }
            )

        # Add action space description
        user_msgs.append({"type": "text", "text": self._get_action_space_description()})

        # Add next action prompt
        user_msgs.append(
            {
                "type": "text",
                "text": "# Next action\nThe task has not been completed yet. You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. The content must be in the same format as shown before in the Action Space. You can plan ahead but only 1 immediate action is needed.",
            }
        )

        return user_msgs

    def _preproc_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def _get_system_prompt(self) -> str:
        return SYSTEM_MINIWOB_PROMPT_WITHOUT_THOUGHT

    def _format_open_tabs(self, urls: list, titles: list, active_index: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"type": "text", "text": "# Currently open tabs (This is the current active tabs)\n"}]

        for idx, (url, title) in enumerate(zip(urls, titles, strict=False)):
            active_marker = " (active tab)" if idx == active_index else ""
            messages.append({"type": "text", "text": f"Tab {idx}{active_marker}\n  Title: {title}\n  URL: {url}\n"})
        return messages

    def _format_screenshot(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        messages.append(
            {
                "type": "text",
                "text": """\
# Current page Screenshot
""",
            }
        )
        messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(screenshot),
                    "detail": "auto",
                },  # Literal["low", "high", "auto"] = "auto"
            }
        )
        return messages

    def _get_action_space_description(self) -> str:
        if self.cot_prompt:
            return f"""\
# Action Space (This is the list of valid actions you are allowed to output after your chain-of-thought reasoning,
{self.action_set.describe(with_long_description=False, with_examples=False)}
Here are examples of actions with chain-of-thought reasoning:
Thought: I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
Action: ```click("12")```
Thought: I found the information requested by the user, I will send it to the chat.
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""
        else:
            return f"""\
# Action Space (This is the list of valid actions you are allowed to output,
{self.action_set.describe(with_long_description=False, with_examples=False)}
Here are examples of actions that can be returned:
Action: ```click("12")```
Action: ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```
"""

    def _format_msgs_as_str(self, msgs: list[dict[str, Any]]) -> str:
        prompt_text_strings = []
        for message in msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append("image_url: " + image_url[:30] + "... (truncated)")
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(f"Unknown message type {repr(message['type'])} in the task goal.")
        return " ".join(prompt_text_strings)

    def _parse_model_response(self, response: str) -> str:
        """
        Extracts the last content enclosed within triple backticks (``` ```) from the response.

        If the response contains multiple segments wrapped in triple backticks,
        this function returns the content of the **last** occurrence.
        If no such formatting is found, it returns the entire response unmodified.

        Args:
            response (str): The raw text response to be processed.

        Returns:
            action (str): The extracted action (content from the last occurrence of triple backticks
                  or the full response if no match is found)
        """
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)  # Find all occurrences
        if matches:
            return matches[-1]
        return response

    def validate_step(self, trajectory_step: dict) -> bool:
        """
        Validates if the trajectory_step(dict) is valid or malformated.
        """
        thought = trajectory_step["thought"]
        action = trajectory_step["action"]

        # Thought and action are the same, meaning the parser didn't work
        if thought == action:
            return False

        # Response has action that results in error
        if trajectory_step["next_observation"]["last_action_error"]:
            return False

        return True
