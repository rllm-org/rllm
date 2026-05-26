"""LaMer (Meta-RL) cumulative Sokoban agent flow.

Runs N sequential episodes on the same puzzle with self-reflection between
failed attempts. Cross-episode reward discounting incentivizes exploration
in early episodes for better exploitation in later ones.

Each episode is a cumulative multi-turn conversation. Reflections are
single-turn trajectories between episodes. All trajectory rewards are
pre-computed with cross-episode discounting before returning.
"""

from __future__ import annotations

import logging
import time

from env_service import LaMerMixin, StepResult, create_env_session, parse_remark
from env_service.sokoban import SokobanEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from ..sokoban_prompt import get_sokoban_prompt, SOKOBAN_PLAY_PROMPT
    from .sokoban_flow import parse_actions, _parse_dim_room, _ACTION_LABELS, _ACTION_COUNT_WORDS
except (ImportError, ValueError):
    from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from sokoban_prompt import get_sokoban_prompt, SOKOBAN_PLAY_PROMPT
    from flow.sokoban_flow import parse_actions, _parse_dim_room, _ACTION_LABELS, _ACTION_COUNT_WORDS

logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3
DEFAULT_TRAJ_GAMMA = 0.6
DEFAULT_REFLECTION_TYPE = "reflection_only"


def _build_system_prompt(actions_per_turn: int) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))
    prompt = SOKOBAN_PLAY_PROMPT.format(
        init_observation="",
        past_trajectories_reflections="\n",
        current_trajectory="",
        num_actions_per_turn=word,
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
    ep_idx: int,
    lamer: LaMerMixin,
    reflection_type: str,
    action_is_valid: bool = True,
) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))

    if turn == 0:
        prompt = get_sokoban_prompt(
            phase="play",
            turn_idx=0,
            traj_idx=ep_idx,
            init_observation=observation,
            curr_traj="",
            past_traj=lamer.past_traj_actions,
            reflection=lamer.reflections,
            num_actions_per_turn=actions_per_turn,
            reflection_type=reflection_type,
        )
        remaining = max(max_turns - turn, 0)
        prompt += f"\n\nYou have {remaining} turns remaining including this one."
        return prompt

    header = f"Current observation (turn {turn}):"
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
        f"Now it's your turn to make moves (choose the next {word} actions)."
    )


@rllm.rollout(name="sokoban_lamer")
async def sokoban_lamer_flow(task: Task, config: AgentConfig) -> Episode:
    """Run N sequential episodes with reflection and cross-episode reward discounting."""
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
            t = time.perf_counter()
            observation, _info = await session.reset()
            env_init_s += time.perf_counter() - t

            if initial_observation is None:
                initial_observation = observation

            initial_prompt = _build_observation_prompt(
                observation=initial_observation,
                turn=0,
                max_turns=max_turns,
                actions_per_turn=actions_per_turn,
                ep_idx=ep_idx,
                lamer=lamer,
                reflection_type=reflection_type,
            )
            messages: list[dict] = [
                {"role": "system", "content": _build_system_prompt(actions_per_turn)},
                {"role": "user", "content": initial_prompt},
            ]

            steps: list[Step] = []
            action_history_lines: list[str] = []
            won = False
            done = False
            ep_env_steps = 0

            for turn in range(max_turns):
                if done or won or ep_env_steps >= max_env_steps:
                    break

                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                except Exception as e:
                    logger.warning(
                        "sokoban_lamer task %s ep %d turn %d: LLM call failed: %s",
                        task.id, ep_idx, turn, e,
                    )
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
                        observation=messages[-2]["content"] if len(messages) >= 2 else "",
                        model_response=content,
                        action=last_action,
                        thought=content,
                    )
                )

                for action in actions:
                    if ep_env_steps >= max_env_steps:
                        break
                    t = time.perf_counter()
                    result = await session.step(action)
                    env_step_s += time.perf_counter() - t
                    ep_env_steps += 1
                    observation = result.observation
                    won = result.won
                    done = result.done
                    if done or won:
                        break

                action_history_lines.append(f"Action {turn + 1}: {last_action}")

                if done or won or ep_env_steps >= max_env_steps:
                    break

                next_prompt = _build_observation_prompt(
                    observation=observation,
                    turn=turn + 1,
                    max_turns=max_turns,
                    actions_per_turn=actions_per_turn,
                    ep_idx=ep_idx,
                    lamer=lamer,
                    reflection_type=reflection_type,
                    action_is_valid=action_is_valid,
                )
                messages.append({"role": "user", "content": next_prompt})

        total_turns += len(steps)
        total_env_steps += ep_env_steps
        ep_reward = 1.0 if won else 0.0
        lamer.record_episode(ep_reward, "\n".join(action_history_lines))

        trajectories.append(Trajectory(name=f"sokoban_ep{ep_idx}", steps=steps, reward=None))

        # --- REFLECT PHASE (only if failed and not last episode) ---
        if not won and ep_idx < num_episodes - 1:
            curr_traj_str = "\n".join(action_history_lines)
            reflect_prompt = get_sokoban_prompt(
                phase="reflect",
                turn_idx=len(steps) - 1 if steps else 0,
                traj_idx=ep_idx,
                init_observation=initial_observation,
                curr_traj=curr_traj_str,
                past_traj=lamer.past_traj_actions,
                reflection=lamer.reflections,
                num_actions_per_turn=actions_per_turn,
                reflection_type=reflection_type,
            )
            reflect_messages = [{"role": "user", "content": reflect_prompt}]

            try:
                resp = await client.chat.completions.create(
                    model=config.model, messages=reflect_messages, **sampling, timeout=120,
                )
                reflect_content = resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "sokoban_lamer task %s ep %d: reflection call failed: %s",
                    task.id, ep_idx, e,
                )
                reflect_content = ""

            remark_text = parse_remark(reflect_content) if reflect_content else ""
            lamer.record_reflection(remark_text)

            reflect_step = Step(
                chat_completions=reflect_messages + [{"role": "assistant", "content": reflect_content}],
                observation=reflect_prompt,
                model_response=reflect_content,
                action="reflect",
                thought=reflect_content,
            )
            trajectories.append(Trajectory(name=f"sokoban_reflect{ep_idx}", steps=[reflect_step], reward=None))

    # --- COMPUTE CROSS-EPISODE DISCOUNTED REWARDS ---
    discounted_rewards = lamer.discounted_rewards

    for traj in trajectories:
        if traj.name.startswith("sokoban_ep"):
            idx = int(traj.name[len("sokoban_ep"):])
            traj.reward = discounted_rewards[idx]
        elif traj.name.startswith("sokoban_reflect"):
            idx = int(traj.name[len("sokoban_reflect"):])
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
            "dim_room": dim_room,
            "num_boxes": num_boxes,
            "max_steps": max_env_steps,
            "max_turns": max_turns,
            "num_episodes": num_episodes,
            "traj_gamma": traj_gamma,
            "mode": mode,
        },
        is_correct=lamer.won_any,
    )
