"""Non-cumulative (step-based) MiniSweeper agent flow.

Each turn constructs a fresh, self-contained prompt with the initial
observation and a sliding window of recent action/observation history.
This produces independent steps that the verl transform handles as
separate training rows — enabling stepwise GRPO with advantage broadcasting.
"""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory
from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
from minisweeper_prompt import get_minesweeper_prompt
from minisweeper_flow import parse_action
from minisweeper_pkg import MineSweeper

logger = logging.getLogger(__name__)

_OBS_LENGTH = 2


def _format_history(history: list[dict], obs_length: int = _OBS_LENGTH) -> str:
    """Format action/obs history with sliding window.

    Recent entries (within obs_length) show full board text.
    Older entries show only truncated placeholders.
    Matches verl-agent SimpleMemoryMineSweeper.fetch() behavior.
    """
    if not history:
        return ""
    lines = []
    for j, rec in enumerate(history):
        step_num = j + 1
        if len(history) - j > obs_length:
            lines.append(f"Action {step_num}: {rec['action']}\nObservation {step_num}: ...")
        else:
            lines.append(f"Action {step_num}: {rec['action']}\nObservation {step_num}:\n{rec['obs']}")
    return "\n".join(lines)


@rllm.rollout(name="minisweeper_step")
async def minisweeper_step_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive a MiniSweeper game with step-independent LLM calls."""
    meta = task.metadata or {}
    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    max_steps = int(meta.get("max_steps", LAMER_MINISWEEPER_CONFIG["max_steps"]))
    max_turns = int(meta.get("max_turns", LAMER_MINISWEEPER_CONFIG["max_turns"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])

    t = time.perf_counter()
    env = MineSweeper(board_size=board_size, n_mines=n_mines, board_type=board_type)
    observation, _info = env.reset(seed=seed)
    initial_observation = observation
    env_init_s = time.perf_counter() - t

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    steps: list[Step] = []
    history: list[dict] = []
    won = False
    done = False
    env_steps = 0
    env_step_s = 0.0

    for turn in range(max_turns):
        if done or won or env_steps >= max_steps:
            break

        curr_traj = _format_history(history, obs_length=_OBS_LENGTH)
        prompt = get_minesweeper_prompt(
            n_mines=n_mines,
            board_size=board_size,
            phase="play",
            turn_idx=turn,
            traj_idx=0,
            init_observation=initial_observation,
            curr_traj=curr_traj,
            past_traj="",
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
            logger.warning("minisweeper_step task %s turn %d: LLM call failed: %s", task.id, turn, e)
            break

        content = resp.choices[0].message.content or ""

        coord = parse_action(content, board_size=board_size)
        if coord is None:
            row, col = -1, -1
            action_str = "invalid"
        else:
            row, col = coord
            action_str = f"({row}, {col})"

        steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": content}],
                observation=prompt,
                model_response=content,
                action=action_str,
                thought=content,
            )
        )

        t = time.perf_counter()
        observation, reward, done, info = env.step("L", row, col)
        env_step_s += time.perf_counter() - t
        env_steps += 1
        won = bool(info.get("won", False))

        history.append({"action": action_str, "obs": observation})

        if done or won or env_steps >= max_steps:
            break

    return Episode(
        trajectories=[Trajectory(name="minisweeper_step", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "turns": len(steps),
            "env_steps": env_steps,
            "board_size": board_size,
            "n_mines": n_mines,
            "max_steps": max_steps,
            "max_turns": max_turns,
            "board_type": board_type,
            "mode": mode,
        },
        is_correct=won,
    )
