"""Multi-turn Sokoban agent flow.

The agent pushes boxes onto targets by emitting one or more movement actions
per turn. The environment loop is self-contained in this plugin and reuses the
LaMer Sokoban game wrapper under ``sokoban_pkg``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence

from env_service import create_env_session
from env_service.sokoban import SokobanEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from ..sokoban_prompt import SOKOBAN_PLAY_PROMPT
except (ImportError, ValueError):
    from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from sokoban_prompt import SOKOBAN_PLAY_PROMPT

logger = logging.getLogger(__name__)


_ACTIONS = {"up": 1, "down": 2, "left": 3, "right": 4}
_ACTION_LABELS = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_GENERIC_BLOCK_RE = re.compile(r"```(?:action)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ACTION_WORD_RE = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)
_ACTION_COUNT_WORDS = {
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
}


def _parse_dim_room(value) -> tuple[int, int]:
    if value is None:
        return (6, 6)
    if isinstance(value, str):
        parts = re.split(r"[x,\s]+", value.strip())
        parts = [p for p in parts if p]
        if len(parts) == 1:
            size = int(parts[0])
            return (size, size)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Invalid Sokoban dim_room: {value!r}")


def _extract_action_text(response: str) -> str | None:
    matches = _ACTION_TAG_RE.findall(response)
    if matches:
        return matches[-1].strip()

    matches = _GENERIC_BLOCK_RE.findall(response)
    if matches:
        return matches[-1].strip()

    return None


def parse_actions(response: str, max_actions: int = 3) -> list[int] | None:
    """Extract Sokoban actions from the last action block in a model response."""
    action_text = _extract_action_text(response)
    if not action_text:
        return None

    actions: list[int] = []
    for match in _ACTION_WORD_RE.finditer(action_text):
        actions.append(_ACTIONS[match.group(1).lower()])
        if len(actions) >= max_actions:
            break

    return actions or None


def _action_count_word(actions_per_turn: int) -> str:
    return _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))


def _build_system_prompt(actions_per_turn: int) -> str:
    """Keep the static Sokoban instructions aligned with the LaMer template."""
    prompt = SOKOBAN_PLAY_PROMPT.format(
        init_observation="",
        past_trajectories_reflections="\n",
        current_trajectory="",
        num_actions_per_turn=_action_count_word(actions_per_turn),
    ).strip()
    rules, sep, observations = prompt.partition("# Observations")
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
    actions_per_turn: int,
    action_is_valid: bool = True,
) -> str:
    header = "The initial state of the game is:" if turn == 0 else f"Current observation (turn {turn}):"
    remaining = max(max_turns - turn, 0)
    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour last response did not contain a valid action, so a no-op was applied. "
            "Choose only from up, down, left, right.\n"
        )
    return (
        f"{header}\n{observation}\n"
        f"{retry}\n"
        f"You have {remaining} turns remaining including this one.\n"
        f"Now it's your turn to make moves (choose the next {_action_count_word(actions_per_turn)} actions)."
    )


@rllm.rollout(name="sokoban")
async def sokoban_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive a Sokoban environment with an LLM until success, failure, or timeout."""
    meta = task.metadata or {}
    seed = int(meta.get("seed", LAMER_SOKOBAN_CONFIG["env_seed"]))
    dim_room = _parse_dim_room(meta.get("dim_room", (meta.get("dim_x", 6), meta.get("dim_y", 6))))
    num_boxes = int(meta.get("num_boxes", LAMER_SOKOBAN_CONFIG["num_boxes"]))
    max_env_steps = int(meta.get("max_steps", LAMER_SOKOBAN_CONFIG["max_steps"]))
    search_depth = int(meta.get("search_depth", LAMER_SOKOBAN_CONFIG["search_depth"]))
    min_steps = int(meta.get("min_steps", LAMER_SOKOBAN_CONFIG["min_steps"]))
    max_sol_steps = int(meta.get("max_sol_steps", LAMER_SOKOBAN_CONFIG["max_sol_steps"]))
    actions_per_turn = max(1, int(meta.get("actions_per_turn", LAMER_SOKOBAN_CONFIG["actions_per_turn"])))
    max_turns = int(meta.get("max_turns", LAMER_SOKOBAN_CONFIG["max_turns"]))
    mode = str(meta.get("mode", LAMER_SOKOBAN_CONFIG["mode"]))
    puzzle_state = meta.get("puzzle_state")

    t = time.perf_counter()
    session = await create_env_session(
        SokobanEnv,
        session_mode="local",
        mode=mode,
        dim_room=dim_room,
        num_boxes=num_boxes,
        max_steps=max_env_steps,
        search_depth=search_depth,
        min_steps=min_steps,
        max_sol_steps=max_sol_steps,
        seed=seed,
        puzzle_state=puzzle_state,
    )

    async with session:
        observation, _info = await session.reset()
        initial_observation = observation
        env_init_s = time.perf_counter() - t

        client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
        initial_prompt = _build_observation_prompt(initial_observation, 0, max_turns, actions_per_turn)
        messages: list[dict] = [
            {"role": "system", "content": _build_system_prompt(actions_per_turn)},
            {"role": "user", "content": initial_prompt},
        ]

        steps: list[Step] = []
        won = False
        done = False
        env_steps = 0
        last_action: str | None = None
        env_step_s = 0.0
        sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

        for turn in range(max_turns):
            if done or won or env_steps >= max_env_steps:
                break

            current_observation_prompt = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else observation
            try:
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
            except Exception as e:
                logger.warning("sokoban task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            actions = parse_actions(content, max_actions=actions_per_turn)
            action_is_valid = actions is not None
            if actions is None:
                actions = [0]
            action_labels = [_ACTION_LABELS[a] for a in actions if a in _ACTION_LABELS]
            last_action = ", ".join(action_labels) if action_labels else "still"

            messages.append({"role": "assistant", "content": content})
            steps.append(
                Step(
                    chat_completions=list(messages),
                    observation=current_observation_prompt,
                    model_response=content,
                    action=last_action,
                    thought=content,
                )
            )

            for action in actions:
                if env_steps >= max_env_steps:
                    break
                t = time.perf_counter()
                result = await session.step(action)
                env_step_s += time.perf_counter() - t
                env_steps += 1
                observation = result.observation
                won = result.won
                done = result.done
                if done or won:
                    break

            if done or won or env_steps >= max_env_steps:
                break
            next_prompt = _build_observation_prompt(
                observation,
                turn + 1,
                max_turns,
                actions_per_turn,
                action_is_valid=action_is_valid,
            )
            messages.append({"role": "user", "content": next_prompt})

    return Episode(
        trajectories=[Trajectory(name="sokoban", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "turns": len(steps),
            "env_steps": env_steps,
            "last_action": last_action,
            "dim_room": dim_room,
            "num_boxes": num_boxes,
            "max_steps": max_env_steps,
            "max_turns": max_turns,
            "mode": mode,
        },
        is_correct=won,
    )
