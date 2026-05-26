"""Non-cumulative (step-based) ALFWorld agent flow.

Each turn constructs a fresh, self-contained prompt with the initial
observation and a sliding window of recent action/observation history.
This produces independent steps that the verl transform handles as
separate training rows -- enabling stepwise GRPO with advantage broadcasting.
"""

from __future__ import annotations

import logging
import os
import time

from env_service import create_env_session
from env_service.alfworld import AlfWorldEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from .alfworld_flow import parse_action
    from ..alfworld_prompt import get_alfworld_prompt
except (ImportError, ValueError):
    from flow.alfworld_flow import parse_action
    from alfworld_prompt import get_alfworld_prompt

logger = logging.getLogger(__name__)

# Sliding window size for history (ALFWorld episodes are short, 15 covers all turns)
_OBS_LENGTH = 15


def _format_history(history: list[dict], obs_length: int = _OBS_LENGTH) -> str:
    """Format action/obs history for prompt construction.

    All entries show: "Action N: {action}\nObservation N: {obs}"
    No truncation needed for alfworld (history_length=15 covers all turns).
    """
    if not history:
        return ""
    lines = []
    for j, rec in enumerate(history):
        step_num = j + 1
        lines.append(f"Action {step_num}: {rec['action']}\nObservation {step_num}: {rec['obs']}")
    return "\n".join(lines)


@rllm.rollout(name="alfworld_step")
async def alfworld_step_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive the ALFWorld TextWorld env with step-independent LLM calls."""
    meta = task.metadata or {}
    game_file = meta.get("game_file")
    if not game_file:
        raise ValueError("Task metadata must include 'game_file'")

    max_steps = int(meta.get("max_steps", 50))
    task_type = meta.get("task_type", "unknown")

    env_step_s = 0.0
    uid = config.session_uid or task.id

    t = time.perf_counter()
    logger.info("alfworld_step rollout %s: env init start game=%s max_steps=%d", uid, os.path.basename(game_file), max_steps)
    session = await create_env_session(AlfWorldEnv, session_mode="ray_pool", game_file=game_file, max_steps=max_steps)
    initial_obs, info = await session.reset()
    admissible_commands = info.get("admissible_commands", [])
    env_init_s = time.perf_counter() - t
    logger.info(
        "alfworld_step rollout %s: env init done in %.2fs admissible_commands=%d",
        uid,
        env_init_s,
        len(admissible_commands),
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    steps: list[Step] = []
    history: list[dict] = []
    won = False
    done = False
    last_action: str | None = None

    async with session:
        for turn in range(max_steps):
            if done or won:
                break

            # Build fresh prompt each turn using sliding-window history
            curr_traj = _format_history(history, obs_length=_OBS_LENGTH)
            admissible_str = ", ".join(f"'{cmd}'" for cmd in admissible_commands)
            prompt = get_alfworld_prompt(
                phase="play",
                turn_idx=turn,
                traj_idx=0,
                init_observation=initial_obs,
                curr_traj=curr_traj,
                past_traj={},
                admissible_actions=admissible_str,
                reflection="",
            )

            messages = [{"role": "user", "content": prompt}]

            try:
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
            except Exception as e:
                logger.warning("alfworld_step task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            action_str = parse_action(content)
            last_action = action_str

            # Non-cumulative: each step's chat_completions is just [user, assistant]
            steps.append(
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": content}],
                    observation=prompt,
                    model_response=content,
                    action=action_str,
                    thought=content,
                )
            )

            if action_str is None:
                history.append({"action": "invalid", "obs": "Your action was not recognized. Please provide a valid action."})
                continue

            # Execute action in environment
            t = time.perf_counter()
            result = await session.step(action_str)
            env_step_s += time.perf_counter() - t

            observation = result.observation
            won = result.won
            done = result.done
            admissible_commands = result.info.get("admissible_commands", [])

            history.append({"action": action_str, "obs": observation})

            if done or won:
                break

    return Episode(
        trajectories=[Trajectory(name="alfworld_step", steps=steps)],
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
