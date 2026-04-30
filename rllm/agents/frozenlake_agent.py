import copy
import logging
import re
from typing import Any

from rllm.agents.agent import BaseAgent
from rllm.agents.system_prompts import *
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.types import Action, Step, Trajectory

logger = logging.getLogger(__name__)


class FrozenLakeAgent(BaseAgent):
    # Prompting format inspired by the RAGEN project: https://github.com/RAGEN-AI/RAGEN
    SYSTEM_PROMPT: str = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    MULTI_SHOT_SYSTEM_PROMPT: str = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Below are examples for an interaction:
Example1:
User: Current Observation:
P   _   _   _   _
O   _   _   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the top right corner. It should reach G at the bottom right corner. I should move it closer to it. I can move right or down but there is a hole in down position and I can not move diagonally. There is no hole in my next movement right so I can move to right. Action: ```Right```

Example2:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   P
O   _   _   G
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the near G. It should reach G to its bottom. I should move to be on it. There is no hole in my next movement so I can move to down. Action: ```Down```

Example3:
User: Current Observation:
_   _   _   O   _
O   _   P   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right relative to P. I want to move closer so I should move right or down. But there is a hole at each position and I do not want to fall into holes. Up and left are both valid but left brings me closer. Action: ```Left```

Example4:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   O
O   G   P   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now near G. But game has not finished. P is not at G and I should never output invalid action. I need to recheck my understanding. P is not actually on G yet because they are not overlapping, it needs reach G to its left. Action: ```Left```

Example5:
User: Current Observation:
_   _   _   O   _
O   _   P   _   _
O   _   O   O   O
O   _   O   G   _
O   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right corner of P. I can move left, right, or up. Move right will initially bring me closer but I can't reach G that way. Move up and left means I can still reach G. Move up will result in 9 steps in total while left is 7 steps. I need to move left. Action: ```Left```

Now it is your turn, please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

    def __init__(self, max_steps: int | None = None, use_accumulate_thinking: bool | None = True, use_multistep_prompt: bool | None = False, use_accumulate_history: bool | None = True):
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step: int = 0
        self.accumulate_thinking: bool | None = use_accumulate_thinking  # controlls whether to accumulate the thinking portion of the response
        self.multistep_prompt: bool | None = use_multistep_prompt
        self.max_steps: int | None = max_steps
        self.accumulate_history: bool | None = use_accumulate_history
        self.current_observation: Any = None
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """

        if self._trajectory.steps:
            cur_step = self._trajectory.steps[-1]
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        current_obs_str = str(observation)
        # Base message for the user
        user_prompt_content = f"Current Observation ({self.step}): \n" + current_obs_str + "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."

        # Check if the observation is the same as the previous step's observation
        # This check only makes sense if we have completed at least one step (i.e., received a model response and acted)
        if self._trajectory.steps and self._trajectory.steps[-1].action is not None:  # Check if the last step has an action (meaning it's a completed step)
            last_step_obs_str = self._trajectory.steps[-1].observation
            if last_step_obs_str == current_obs_str:
                user_prompt_content += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."

        if self.max_steps is not None and self.max_steps - self.step > 0:
            user_prompt_content += f"\nThe maximum number of steps remaining is {self.max_steps - self.step}."

        # Add the user message for the *next* interaction turn
        self.messages.append({"role": "user", "content": user_prompt_content})

        self.current_observation = current_obs_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        content = response

        if not self.accumulate_thinking:
            _, sep, after = content.partition("</think>")
            if sep:
                content = after

        thought, action_str = self._parse_model_response(content)

        self.messages.append({"role": "assistant", "content": content})

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), thought=thought, action=action_str, model_response=content, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        self.step += 1

        return Action(action=action_str)

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = response
        action_str = str(FrozenLakeEnv.INVALID_ACTION)

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = response.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = response[:last_match_index].strip()

            extracted_text = last_match_content.lower()

            if extracted_text in DIRECTION_MAP:
                action_str = str(DIRECTION_MAP[extracted_text])
            elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
                action_str = str(int(extracted_text))

        return thought, action_str

    def _process_action_for_validation(self, response: str) -> str:
        _, action_str = self._parse_model_response(response)
        return action_str

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self.accumulate_history:
            return self.messages
        else:
            if len(self.messages) <= 1:
                return self.messages
            else:
                return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT if not self.multistep_prompt else self.MULTI_SHOT_SYSTEM_PROMPT,
            }
        ]
        self.step = 0
