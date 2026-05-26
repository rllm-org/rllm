"""LaMer (Meta-RL) cumulative MineSweeper agent flow.

Runs N sequential episodes on the same board with self-reflection between
failed attempts. Cross-episode reward discounting incentivizes exploration
in early episodes for better exploitation in later ones.
"""

from __future__ import annotations

import logging
import time

from env_service import LaMerMixin, StepResult, create_env_session, parse_remark
from env_service.minesweeper import MineSweeperEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..minisweeper_prompt import get_minesweeper_prompt
    from ..prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from .minisweeper_flow import parse_action, _build_system_prompt, _build_observation_prompt
except (ImportError, ValueError):
    from minisweeper_prompt import get_minesweeper_prompt
    from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from flow.minisweeper_flow import parse_action, _build_system_prompt, _build_observation_prompt

logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3
DEFAULT_TRAJ_GAMMA = 0.6
DEFAULT_REFLECTION_TYPE = "reflection_only"


@rllm.rollout(name="minisweeper_lamer")
async def minisweeper_lamer_flow(task: Task, config: AgentConfig) -> Episode:
    """Run N sequential MineSweeper episodes with reflection and cross-episode rewards."""
    meta = task.metadata or {}

    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    max_steps = int(meta.get("max_steps", LAMER_MINISWEEPER_CONFIG["max_steps"]))
    max_turns = int(meta.get("max_turns", LAMER_MINISWEEPER_CONFIG["max_turns"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])
    puzzle_state = meta.get("puzzle_state")

    num_episodes = int(meta.get("num_episodes", DEFAULT_NUM_EPISODES))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))
    reflection_type = str(meta.get("reflection_type", DEFAULT_REFLECTION_TYPE))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    lamer = LaMerMixin(num_episodes=num_episodes, traj_gamma=traj_gamma)
    trajectories: list[Trajectory] = []
    initial_observation: str | None = None
    total_turns = 0
    total_env_steps = 0
    env_init_s = 0.0
    env_step_s = 0.0

    for ep_idx in range(num_episodes):
        if not lamer.should_continue and ep_idx > 0:
            break

        # --- PLAY PHASE ---
        session = await create_env_session(
            MineSweeperEnv,
            session_mode="local",
            board_size=board_size,
            n_mines=n_mines,
            board_type=board_type,
            seed=seed,
            puzzle_state=puzzle_state,
        )
        async with session:
            t = time.perf_counter()
            observation, _info = await session.reset()
            env_init_s += time.perf_counter() - t

            if initial_observation is None:
                initial_observation = observation

            if ep_idx == 0:
                initial_prompt = _build_observation_prompt(initial_observation, 0, max_turns)
            else:
                initial_prompt = get_minesweeper_prompt(
                    n_mines=n_mines,
                    board_size=board_size,
                    phase="play",
                    turn_idx=0,
                    traj_idx=ep_idx,
                    init_observation=initial_observation,
                    curr_traj="",
                    past_traj=lamer.past_traj_actions,
                    reflection=lamer.reflections,
                    reflection_type=reflection_type,
                )
                remaining = max(max_turns, 0)
                initial_prompt += f"\n\nYou have {remaining} turns remaining including this one."

            messages: list[dict] = [
                {"role": "system", "content": _build_system_prompt(board_size, n_mines)},
                {"role": "user", "content": initial_prompt},
            ]

            steps: list[Step] = []
            action_history_lines: list[str] = []
            won = False
            done = False
            ep_env_steps = 0

            for turn in range(max_turns):
                if done or won or ep_env_steps >= max_steps:
                    break

                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                except Exception as e:
                    logger.warning("minisweeper_lamer task %s ep %d turn %d: LLM failed: %s", task.id, ep_idx, turn, e)
                    break

                content = resp.choices[0].message.content or ""
                action = parse_action(content, board_size=board_size)
                action_is_valid = action is not None
                env_action = action if action is not None else (-1, -1)

                messages.append({"role": "assistant", "content": content})
                steps.append(
                    Step(
                        chat_completions=list(messages),
                        model_response=content,
                        action=f"({env_action[0]}, {env_action[1]})" if action else None,
                        thought=content,
                    )
                )

                t = time.perf_counter()
                result = await session.step(("L", env_action[0], env_action[1]))
                env_step_s += time.perf_counter() - t
                ep_env_steps += 1

                observation = result.observation
                done = result.done
                won = result.won

                action_history_lines.append(
                    f"Action {turn + 1}: ({env_action[0]}, {env_action[1]})"
                )

                if done or won or ep_env_steps >= max_steps:
                    break

                messages.append({
                    "role": "user",
                    "content": _build_observation_prompt(observation, turn + 1, max_turns, action_is_valid),
                })

        total_turns += len(steps)
        total_env_steps += ep_env_steps
        ep_reward = 1.0 if won else 0.0
        lamer.record_episode(ep_reward, "\n".join(action_history_lines))

        trajectories.append(Trajectory(name=f"minisweeper_ep{ep_idx}", steps=steps, reward=None))

        # --- REFLECT PHASE ---
        if not won and ep_idx < num_episodes - 1:
            curr_traj_str = "\n".join(action_history_lines)
            reflect_prompt = get_minesweeper_prompt(
                n_mines=n_mines,
                board_size=board_size,
                phase="reflect",
                turn_idx=len(steps) - 1 if steps else 0,
                traj_idx=ep_idx,
                init_observation=initial_observation,
                curr_traj=curr_traj_str,
                past_traj=lamer.past_traj_actions,
                reflection=lamer.reflections,
                reflection_type=reflection_type,
            )
            reflect_messages = [{"role": "user", "content": reflect_prompt}]

            try:
                resp = await client.chat.completions.create(
                    model=config.model, messages=reflect_messages, **sampling, timeout=120,
                )
                reflect_content = resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning("minisweeper_lamer task %s ep %d: reflect failed: %s", task.id, ep_idx, e)
                reflect_content = ""

            remark_text = parse_remark(reflect_content) if reflect_content else ""
            lamer.record_reflection(remark_text)

            reflect_step = Step(
                chat_completions=reflect_messages + [{"role": "assistant", "content": reflect_content}],
                model_response=reflect_content,
                action="reflect",
                thought=reflect_content,
            )
            trajectories.append(Trajectory(name=f"minisweeper_reflect{ep_idx}", steps=[reflect_step], reward=None))

    # --- COMPUTE CROSS-EPISODE DISCOUNTED REWARDS ---
    discounted_rewards = lamer.discounted_rewards

    for traj in trajectories:
        if traj.name.startswith("minisweeper_ep"):
            idx = int(traj.name[len("minisweeper_ep"):])
            traj.reward = discounted_rewards[idx]
        elif traj.name.startswith("minisweeper_reflect"):
            idx = int(traj.name[len("minisweeper_reflect"):])
            traj.reward = discounted_rewards[idx + 1]

    return Episode(
        trajectories=trajectories,
        metrics={"time/env_init_s": env_init_s, "time/env_step_s": env_step_s},
        artifacts={
            "won": lamer.won_any,
            "episodes_played": len(lamer.episode_rewards),
            "episode_rewards": lamer.episode_rewards,
            "discounted_rewards": discounted_rewards,
            "turns_total": total_turns,
            "env_steps_total": total_env_steps,
            "board_size": board_size,
            "n_mines": n_mines,
            "num_episodes": num_episodes,
            "traj_gamma": traj_gamma,
        },
        is_correct=lamer.won_any,
    )
