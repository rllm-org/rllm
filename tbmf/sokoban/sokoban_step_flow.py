"""Non-cumulative (step-based) Sokoban agent flow.

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
from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
from sokoban_prompt import get_sokoban_prompt
from sokoban_flow import parse_actions, _parse_dim_room, _ACTION_LABELS
from sokoban_pkg import SokobanEnv

logger = logging.getLogger(__name__)

_OBS_LENGTH = 2


def _format_history(history: list[dict], obs_length: int = _OBS_LENGTH) -> str:
    """Format action/obs history with sliding window.

    Recent entries (within obs_length) show full grid text.
    Older entries show only truncated placeholders.
    Matches verl-agent SimpleMemorySokoban.fetch() behavior.
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


@rllm.rollout(name="sokoban_step")
async def sokoban_step_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive a Sokoban environment with step-independent LLM calls."""
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

    t = time.perf_counter()
    env = SokobanEnv(
        mode=mode,
        dim_room=dim_room,
        num_boxes=num_boxes,
        max_steps=max_env_steps,
        search_depth=search_depth,
        min_steps=min_steps,
        max_sol_steps=max_sol_steps,
    )
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
    last_action: str | None = None
    env_step_s = 0.0

    try:
        for turn in range(max_turns):
            if done or won or env_steps >= max_env_steps:
                break

            curr_traj = _format_history(history, obs_length=_OBS_LENGTH)
            prompt = get_sokoban_prompt(
                phase="play",
                turn_idx=turn,
                traj_idx=0,
                init_observation=initial_observation,
                curr_traj=curr_traj,
                past_traj={},
                reflection="",
                num_actions_per_turn=actions_per_turn,
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
                logger.warning("sokoban_step task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            actions = parse_actions(content, max_actions=actions_per_turn)
            if actions is None:
                actions = [0]
            action_labels = [_ACTION_LABELS[a] for a in actions if a in _ACTION_LABELS]
            last_action = ", ".join(action_labels) if action_labels else "still"

            steps.append(
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": content}],
                    observation=prompt,
                    model_response=content,
                    action=last_action,
                    thought=content,
                )
            )

            for action in actions:
                if env_steps >= max_env_steps:
                    break
                t = time.perf_counter()
                observation, _, done, info = env.step(action)
                env_step_s += time.perf_counter() - t
                env_steps += 1
                won = bool(info.get("won", False))
                if done or won:
                    break

            history.append({"action": last_action, "obs": observation})

            if done or won or env_steps >= max_env_steps:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return Episode(
        trajectories=[Trajectory(name="sokoban_step", steps=steps)],
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
