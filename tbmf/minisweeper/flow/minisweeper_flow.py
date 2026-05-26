"""Multi-turn MiniSweeper agent flow.

The agent plays a compact Minesweeper board by revealing one unopened cell per
turn. The game engine is vendored from the LaMer MiniSweeper environment under
``minisweeper_pkg``.
"""

from __future__ import annotations

import logging
import re
import time

from env_service import create_env_session
from env_service.minesweeper import MineSweeperEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..minisweeper_prompt import MINESWEEPER_PLAY_PROMPT
    from ..prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
except (ImportError, ValueError):
    from minisweeper_prompt import MINESWEEPER_PLAY_PROMPT
    from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG

logger = logging.getLogger(__name__)


_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_GENERIC_BLOCK_RE = re.compile(r"```(?:action)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_COORD_RE = re.compile(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")


def _extract_action_text(response: str) -> str | None:
    matches = _ACTION_TAG_RE.findall(response)
    if matches:
        return matches[-1].strip()

    matches = _GENERIC_BLOCK_RE.findall(response)
    if matches:
        return matches[-1].strip()

    return None


def parse_action(response: str, board_size: int | None = None) -> tuple[int, int] | None:
    """Extract a 1-indexed ``(row, col)`` reveal action from a model response."""
    action_text = _extract_action_text(response)
    if not action_text:
        return None

    matches = _COORD_RE.findall(action_text)
    if not matches:
        return None

    row, col = (int(v) for v in matches[-1])
    if board_size is not None and not (1 <= row <= board_size and 1 <= col <= board_size):
        return None
    return row, col


def _build_system_prompt(board_size: int, n_mines: int) -> str:
    """Keep the static MiniSweeper instructions aligned with the LaMer template."""
    prompt = MINESWEEPER_PLAY_PROMPT.format(
        board_size=board_size,
        n_mines=n_mines,
        init_observation="",
        past_trajectories_reflections="\n",
        current_trajectory="",
    ).strip()
    rules, sep, observations = prompt.partition("# Observation")
    if not sep:
        return prompt
    _obs, turn_sep, response_format = observations.partition("Now it's your turn")
    if not turn_sep:
        return rules.strip()
    return f"{rules.strip()}\n\n# Response Format\nNow it's your turn{response_format.rstrip()}"


def _build_observation_prompt(
    observation: str,
    turn: int,
    max_turns: int,
    action_is_valid: bool = True,
) -> str:
    header = "The initial state of the game is:" if turn == 0 else f"Current observation (turn {turn}):"
    remaining = max(max_turns - turn, 0)
    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour last response did not contain a valid coordinate, so an invalid reveal was applied. "
            "Choose one unopened cell in the format (row, col).\n"
        )
    return (
        f"{header}\n{observation}\n"
        f"{retry}\n"
        f"You have {remaining} turns remaining including this one.\n"
        "Now it's your turn to make a move."
    )


@rllm.rollout(name="minisweeper")
async def minisweeper_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive a MiniSweeper game with an LLM until win, mine hit, or timeout."""
    meta = task.metadata or {}
    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    max_steps = int(meta.get("max_steps", LAMER_MINISWEEPER_CONFIG["max_steps"]))
    max_turns = int(meta.get("max_turns", LAMER_MINISWEEPER_CONFIG["max_turns"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])
    puzzle_state = meta.get("puzzle_state")

    t = time.perf_counter()
    session = await create_env_session(
        MineSweeperEnv,
        session_mode="local",
        board_size=board_size,
        n_mines=n_mines,
        board_type=board_type,
        seed=seed,
        puzzle_state=puzzle_state,
    )
    observation, _info = await session.reset()
    initial_observation = observation
    env_init_s = time.perf_counter() - t

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(board_size, n_mines)},
        {"role": "user", "content": _build_observation_prompt(initial_observation, 0, max_turns)},
    ]

    steps: list[Step] = []
    won = False
    done = False
    env_steps = 0
    last_action: tuple[int, int] | None = None
    env_step_s = 0.0
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    async with session:
        for turn in range(max_turns):
            if done or won or env_steps >= max_steps:
                break

            try:
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
            except Exception as e:
                logger.warning("minisweeper task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            action = parse_action(content, board_size=board_size)
            action_is_valid = action is not None
            env_action = action if action is not None else (-1, -1)
            last_action = action

            messages.append({"role": "assistant", "content": content})
            steps.append(
                Step(
                    chat_completions=list(messages),
                    model_response=content,
                    action=f"({action[0]}, {action[1]})" if action else None,
                    thought=content,
                )
            )

            t = time.perf_counter()
            result = await session.step(("L", env_action[0], env_action[1]))
            env_step_s += time.perf_counter() - t
            env_steps += 1

            observation = result.observation
            won = result.won
            done = result.done

            if done or won or env_steps >= max_steps:
                break

            messages.append({
                "role": "user",
                "content": _build_observation_prompt(
                    observation, turn + 1, max_turns, action_is_valid=action_is_valid,
                ),
            })

    return Episode(
        trajectories=[Trajectory(name="minisweeper", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "turns": len(steps),
            "env_steps": env_steps,
            "last_action": last_action,
            "board_size": board_size,
            "n_mines": n_mines,
            "max_steps": max_steps,
            "max_turns": max_turns,
            "board_type": board_type,
            "mode": mode,
        },
        is_correct=won,
    )
