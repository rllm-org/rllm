"""Multi-turn ALFWorld agent flow.

The agent performs household tasks (pick-and-place, heat, cool, clean, etc.)
in a text-based environment by emitting natural-language commands.

Task metadata schema (see ``prepare_alfworld_data.py``)::

    {"game_file": str, "task_type": str, "task_id": str, "max_steps": int}

The game file path points to a .tw-pddl file which TextWorld loads to
instantiate the environment.
"""

from __future__ import annotations

import logging
import os
import re
import time

from env_service import create_env_session
from env_service.alfworld import AlfWorldEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful assistant controlling a robot in a household environment. Your task is to complete household tasks by navigating rooms and interacting with objects.

Action Commands:
Your <action> must be one of the following, strictly following the command (argument) format.

## Navigation & Observation:
- look: Look around your current location to get more details.
- inventory: Check the object you are currently holding (you can only hold one).
- go to (receptacle): Move to a receptacle (e.g., table, fridge, sink).

## Interacting with Receptacles:
- open (receptacle): Open a receptacle.
- close (receptacle): Close a receptacle.

## Interacting with Objects:
- take (object) from (receptacle): Pick up an object from a receptacle.
- put (object) in/on (receptacle): Place the object you are holding into or onto a receptacle.
- examine (object): Examine an object closely to learn its properties.

## Changing Object States:
- heat (object) with (receptacle): Heat an object with a device (e.g., microwave).
- cool (object) with (receptacle): Cool an object with a device (e.g., fridge).
- clean (object) with (receptacle): Clean an object with a device (e.g., sink).
- slice (object) with (object): Slice an object using a sharp object (e.g., knife).

Important Rules
1. You must first navigate to a location before interacting with objects there
2. You can only hold one object at a time
3. Some objects need to be heated, cooled, or cleaned before placing them
4. Always check admissible commands to see what actions are currently valid

Response Format
Think through your approach step by step, then provide your action in the following format:
<think>
Your reasoning and analysis here...
</think>
```action
<your action here>
```

you must adhere to the response format strictly or your action will not be executed:
For example:
<think>
I need to go to the countertop to get the knife.
</think>
```action
go to countertop 1
```

Always provide exactly ONE action at a time.
"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"```action\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_GENERIC_RE = re.compile(r"```(?!action\b)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_action(response: str) -> str | None:
    """Extract action string from the model response.

    Looks for ```action ... ``` blocks first, then falls back to generic ``` ... ```.
    Returns None if no valid action block is found.
    """
    matches = _ACTION_RE.findall(response)
    if matches:
        action = matches[-1].strip()
        if action:
            return action

    matches = _GENERIC_RE.findall(response)
    if matches:
        action = matches[-1].strip()
        if action and not action.startswith("<") and "\n" not in action:
            return action

    return None


# ---------------------------------------------------------------------------
# AgentFlow
# ---------------------------------------------------------------------------


@rllm.rollout(name="alfworld")
async def alfworld_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive the ALFWorld TextWorld env with an LLM until termination."""
    meta = task.metadata or {}
    game_file = meta.get("game_file")
    if not game_file:
        raise ValueError("Task metadata must include 'game_file'")

    max_steps = int(meta.get("max_steps", 50))
    task_type = meta.get("task_type", "unknown")
    uid = config.session_uid or task.id
    env_step_s = 0.0

    t = time.perf_counter()
    logger.info("alfworld rollout %s: env init start game=%s max_steps=%d", uid, os.path.basename(game_file), max_steps)
    session = await create_env_session(AlfWorldEnv, session_mode="ray_pool", game_file=game_file, max_steps=max_steps)
    initial_obs, info = await session.reset()
    admissible_commands = info.get("admissible_commands", [])
    env_init_s = time.perf_counter() - t
    logger.info("alfworld rollout %s: env init done in %.2fs admissible_commands=%d", uid, env_init_s, len(admissible_commands))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

    commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)
    initial_prompt = (
        f"Observation:\n{initial_obs}\n\n"
        f"Admissible commands:\n{commands_str}\n\n"
        f"Steps remaining: {max_steps}\n\n"
        f"What is your next action?"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt},
    ]

    steps: list[Step] = []
    won = False
    last_action: str | None = None
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    async with session:
        for turn in range(max_steps):
            try:
                if turn == 0:
                    logger.info("alfworld rollout %s: first LLM request start", uid)
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
                if turn == 0:
                    logger.info("alfworld rollout %s: first LLM response received", uid)
            except Exception as e:
                logger.warning("alfworld task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            action_str = parse_action(content)
            last_action = action_str

            messages.append({"role": "assistant", "content": content})
            steps.append(
                Step(
                    chat_completions=list(messages),
                    model_response=content,
                    action=action_str,
                    thought=content,
                )
            )

            if action_str is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response did not contain a valid action. "
                        "Please reply with your action inside ```action ... ``` blocks.\n\n"
                        f"Admissible commands:\n{commands_str}\n\n"
                        "What is your next action?"
                    ),
                })
                continue

            t = time.perf_counter()
            result = await session.step(action_str)
            env_step_s += time.perf_counter() - t

            observation = result.observation
            won = result.won
            done = result.done
            admissible_commands = result.info.get("admissible_commands", [])

            if done:
                break

            commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)
            remaining = max_steps - (turn + 1)
            user_msg = f"Observation:\n{observation}\n"

            if admissible_commands:
                user_msg += f"\nAdmissible commands:\n{commands_str}\n"

            user_msg += f"\nSteps remaining: {remaining}"

            if "Nothing happens" in observation:
                user_msg += f"\nNote: Your last action '{action_str}' had no effect. Please try a different action."

            user_msg += "\n\nWhat is your next action?"
            messages.append({"role": "user", "content": user_msg})

    return Episode(
        trajectories=[Trajectory(name="alfworld", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "turns": len(steps),
            "last_action": last_action,
            "task_type": task_type,
            "env_backend": "ray",
        },
        is_correct=won,
    )
