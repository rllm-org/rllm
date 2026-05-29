"""LaMer (Meta-RL) cumulative WebShop agent flow.

Runs N sequential episodes on the same shopping task with self-reflection
between failed attempts. Cross-episode reward discounting incentivizes
exploration in early episodes.
"""

from __future__ import annotations

import logging
import re
import time

from env_service import LaMerMixin, create_env_session, parse_remark
from env_service.webshop import WebShopEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from .webshop_flow import (
        parse_action, _build_system_prompt, _build_user_prompt,
        _strip_think_block, _valid_action, load_world_model_summary,
    )
except (ImportError, ValueError):
    from prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from flow.webshop_flow import (
        parse_action, _build_system_prompt, _build_user_prompt,
        _strip_think_block, _valid_action, load_world_model_summary,
    )

logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3
DEFAULT_TRAJ_GAMMA = 0.6
DEFAULT_REFLECTION_TYPE = "reflection_only"

WEBSHOP_REFLECT_PROMPT = '''You are a helpful shopping assistant navigating an e-commerce website.

# Your Task
You will be given the history of a past shopping attempt.
Your job is to **reflect on the past experience**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** for your next attempt.

# Shopping Task
{task_description}

# Past Experience
{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Your response should first analyze why the previous attempt failed (wrong product, wrong options, inefficient navigation, etc.).
- Then devise a concise, new strategy that accounts for your mistakes.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags.
'''

REFLECTION_ONLY_TEMPLATE = '''
On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}'''


def _format_reflections(reflections: list[str]) -> str:
    if not reflections:
        return ""
    parts = []
    for idx, ref in enumerate(reflections):
        parts.append(REFLECTION_ONLY_TEMPLATE.format(traj_idx=idx + 1, reflection=ref))
    return "".join(parts)


@rllm.rollout(name="webshop_lamer")
async def webshop_lamer_flow(task: Task, config: AgentConfig) -> Episode:
    """Run N sequential WebShop episodes with reflection and cross-episode rewards."""
    meta = task.metadata or {}

    session_id = int(meta.get("session_id"))
    max_steps = int(meta.get("max_steps", LAMER_WEBSHOP_CONFIG["max_steps"]))
    max_turns = int(meta.get("max_turns", max_steps))
    seed = int(meta.get("seed", LAMER_WEBSHOP_CONFIG["env_seed"]))
    observation_mode = str(meta.get("observation_mode", LAMER_WEBSHOP_CONFIG["observation_mode"]))
    num_products = meta.get("num_products", LAMER_WEBSHOP_CONFIG["num_products"])
    human_goals = bool(meta.get("human_goals", LAMER_WEBSHOP_CONFIG["human_goals"]))
    use_available_actions = bool(meta.get("use_available_actions", True))
    use_accumulate_history = bool(meta.get("use_accumulate_history", True))
    use_accumulate_thinking = bool(meta.get("use_accumulate_thinking", True))
    file_path = meta.get("file_path")
    attr_path = meta.get("attr_path")
    world_model_summary = load_world_model_summary(meta.get("world_model_file"))

    num_episodes = int(meta.get("num_episodes", DEFAULT_NUM_EPISODES))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))
    reflection_type = str(meta.get("reflection_type", DEFAULT_REFLECTION_TYPE))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    lamer = LaMerMixin(num_episodes=num_episodes, traj_gamma=traj_gamma)
    trajectories: list[Trajectory] = []
    total_turns = 0
    total_env_steps = 0
    env_init_s = 0.0
    env_step_s = 0.0
    current_instruction = ""

    for ep_idx in range(num_episodes):
        if not lamer.should_continue and ep_idx > 0:
            break

        # --- PLAY PHASE ---
        session = await create_env_session(
            WebShopEnv,
            session_mode="ray_pool",
            observation_mode=observation_mode,
            max_steps=max_steps,
            num_products=num_products,
            session_id=session_id,
            seed=seed,
            file_path=file_path,
            attr_path=attr_path,
            human_goals=human_goals,
        )

        t = time.perf_counter()
        observation, info = await session.reset()
        env_init_s += time.perf_counter() - t
        available_actions = info.get("available_actions", {})
        current_instruction = info.get("instruction", task.instruction if isinstance(task.instruction, str) else "")

        system_prompt = _build_system_prompt(world_model_summary)
        # Inject reflection context for ep > 0
        if ep_idx > 0 and lamer.reflections:
            reflection_context = _format_reflections(lamer.reflections)
            system_prompt += f"\n\n## Past Attempt Reflections\n{reflection_context}\n\nNow you're on trial #{ep_idx + 1}. Use these reflections to improve your approach."

        current_prompt = _build_user_prompt(
            observation, current_instruction, available_actions,
            turn=0, max_turns=max_turns,
            use_available_actions=use_available_actions, action_is_valid=True,
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_prompt},
        ]

        steps: list[Step] = []
        action_history_lines: list[str] = []
        ep_env_steps = 0
        won = False

        async with session:
            for turn in range(max_turns):
                if ep_env_steps >= max_steps:
                    break

                model_messages = messages if use_accumulate_history else [messages[0], messages[-1]]
                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=model_messages, **sampling, timeout=120,
                    )
                except Exception as e:
                    logger.warning("webshop_lamer task %s ep %d turn %d: LLM failed: %s", task.id, ep_idx, turn, e)
                    break

                raw_content = resp.choices[0].message.content or ""
                assistant_content = raw_content if use_accumulate_thinking else _strip_think_block(raw_content)
                action = parse_action(raw_content, has_search_bar=bool(available_actions.get("has_search_bar", False)))
                action_is_valid = _valid_action(action, info.get("available_actions", {}))
                if action is None:
                    action = "click[back to search]"

                step_messages = list(model_messages) + [{"role": "assistant", "content": assistant_content}]

                t = time.perf_counter()
                result = await session.step(action)
                env_step_s += time.perf_counter() - t
                ep_env_steps += 1
                observation = result.observation
                done = result.done
                won = result.won
                available_actions = result.info.get("available_actions", {})

                steps.append(
                    Step(
                        chat_completions=step_messages,
                        model_response=assistant_content,
                        action=action,
                        thought=assistant_content,
                    )
                )
                action_history_lines.append(f"Action {turn + 1}: {action}")

                if done or won or ep_env_steps >= max_steps:
                    break

                if use_accumulate_history:
                    messages = list(step_messages)
                else:
                    messages = [messages[0]]

                current_prompt = _build_user_prompt(
                    observation, current_instruction, available_actions,
                    turn=turn + 1, max_turns=max_turns,
                    use_available_actions=use_available_actions, action_is_valid=action_is_valid,
                )
                messages.append({"role": "user", "content": current_prompt})

        total_turns += len(steps)
        total_env_steps += ep_env_steps
        ep_reward = 1.0 if won else 0.0
        lamer.record_episode(ep_reward, "\n".join(action_history_lines))

        trajectories.append(Trajectory(name=f"webshop_ep{ep_idx}", steps=steps, reward=None))

        # --- REFLECT PHASE ---
        if not won and ep_idx < num_episodes - 1:
            curr_traj_str = "\n".join(action_history_lines)
            reflect_prompt = WEBSHOP_REFLECT_PROMPT.format(
                task_description=current_instruction,
                current_trajectory=curr_traj_str,
            )
            reflect_messages = [{"role": "user", "content": reflect_prompt}]

            try:
                resp = await client.chat.completions.create(
                    model=config.model, messages=reflect_messages, **sampling, timeout=120,
                )
                reflect_content = resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning("webshop_lamer task %s ep %d: reflect failed: %s", task.id, ep_idx, e)
                reflect_content = ""

            remark_text = parse_remark(reflect_content) if reflect_content else ""
            lamer.record_reflection(remark_text)

            reflect_step = Step(
                chat_completions=reflect_messages + [{"role": "assistant", "content": reflect_content}],
                model_response=reflect_content,
                action="reflect",
                thought=reflect_content,
            )
            trajectories.append(Trajectory(name=f"webshop_reflect{ep_idx}", steps=[reflect_step], reward=None))

    # --- COMPUTE CROSS-EPISODE DISCOUNTED REWARDS ---
    discounted_rewards = lamer.discounted_rewards

    for traj in trajectories:
        if traj.name.startswith("webshop_ep"):
            idx = int(traj.name[len("webshop_ep"):])
            traj.reward = discounted_rewards[idx]
        elif traj.name.startswith("webshop_reflect"):
            idx = int(traj.name[len("webshop_reflect"):])
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
            "session_id": session_id,
            "instruction": current_instruction,
            "num_episodes": num_episodes,
            "traj_gamma": traj_gamma,
        },
        is_correct=lamer.won_any,
    )
