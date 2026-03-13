"""FrozenLake AgentFlow — multi-turn grid navigation agent."""

from __future__ import annotations

import logging
import re

import openai

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory

from .env import ACTION_INVALID, FrozenLakeEnv

logger = logging.getLogger(__name__)

DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

SYSTEM_PROMPT = """\
You are a helpful assistant. You are walking on a frozen lake.

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

DEFAULT_MAX_STEPS = 10


def _parse_action(response: str) -> int:
    """Extract a direction action from the model response.

    Looks for the last ```...``` block and maps its content to an action int.
    Returns ACTION_INVALID (0) if parsing fails.
    """
    matches = re.findall(r"```(.*?)```", response, re.DOTALL)
    if not matches:
        return ACTION_INVALID

    text = matches[-1].strip().lower()
    if text in DIRECTION_MAP:
        return DIRECTION_MAP[text]
    if text.isdigit() and int(text) in DIRECTION_MAP.values():
        return int(text)
    return ACTION_INVALID


class FrozenLakeAgentFlow:
    """AgentFlow implementation for the FrozenLake grid navigation task."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        seed = task.get("seed", 42)
        size = task.get("size", 4)
        p = task.get("p", 0.8)
        max_steps = task.get("max_steps", DEFAULT_MAX_STEPS)

        env = FrozenLakeEnv(size=size, p=p, seed=seed, max_steps=max_steps)
        obs = env.reset()

        client = openai.OpenAI(base_url=config.base_url, api_key="not-needed")

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        steps: list[Step] = []
        num_steps = 0

        for turn in range(max_steps):
            user_content = f"Current Observation ({turn}):\n{obs}\nYou have not achieved the goal, P has not reached G yet. Please give the next action."
            if turn > 0 and steps and not steps[-1].metadata.get("action_is_effective", True):
                user_content += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."
            remaining = max_steps - turn
            user_content += f"\nThe maximum number of steps remaining is {remaining}."

            messages.append({"role": "user", "content": user_content})

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=0.0,
            )
            assistant_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            action = _parse_action(assistant_text)
            obs, reward, done, info = env.step(action)
            num_steps += 1

            steps.append(
                Step(
                    input=user_content,
                    output=assistant_text,
                    action=action,
                    reward=reward,
                    done=done,
                    metadata=info,
                )
            )

            if done:
                break

        success = env.success()
        task_id = task.get("task_id", f"frozenlake_s{seed}")

        trajectory = Trajectory(
            name="navigator",
            task=task,
            steps=steps,
            reward=1.0 if success else 0.0,
        )

        return Episode(
            id=f"{task_id}:0",
            task=task,
            trajectories=[trajectory],
            artifacts={"success": success, "num_steps": num_steps},
        )


# Module-level singleton for plugin entry point
frozenlake_agent = FrozenLakeAgentFlow()
