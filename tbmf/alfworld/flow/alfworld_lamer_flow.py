"""LaMer (Meta-RL) cumulative ALFWorld agent flow.

Runs N sequential episodes on the same household task with self-reflection
between failed attempts. Cross-episode reward discounting incentivizes
exploration in early episodes.

Each episode creates a fresh Ray-based TextWorld env session (same game_file
= same task layout). The agent accumulates reflection context across episodes.
"""

from __future__ import annotations

import logging
import time

from env_service import LaMerMixin, create_env_session, parse_remark
from env_service.alfworld import AlfWorldEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from .alfworld_flow import parse_action, SYSTEM_PROMPT
except (ImportError, ValueError):
    from flow.alfworld_flow import parse_action, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_NUM_EPISODES = 3
DEFAULT_TRAJ_GAMMA = 0.6

ALFWORLD_REFLECT_PROMPT = '''You are a helpful assistant controlling a robot in a household environment.

# Your Task
You will be given the history of a past attempt at completing a household task.
Your job is to **reflect on the past experience**, identify any **mistakes or inefficiencies**, and then devise a **concise, improved plan** for your next attempt.

# Past Experience
Initial observation:
{init_observation}

Actions taken:
{current_trajectory}
The task is NOT successfully completed.

Now it's your turn to reflect on the past experience and come up with a new plan of action.

- Analyze why the previous attempt failed (wrong location, wrong object, missed step, etc.).
- Devise a concise, new strategy that accounts for your mistakes.
- End the response with your reflection and improved plan inside <remark> </remark> tags.
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


@rllm.rollout(name="alfworld_lamer")
async def alfworld_lamer_flow(task: Task, config: AgentConfig) -> Episode:
    """Run N sequential ALFWorld episodes with reflection and cross-episode rewards."""
    meta = task.metadata or {}
    game_file = meta.get("game_file")
    if not game_file:
        raise ValueError("Task metadata must include 'game_file'")

    max_steps = int(meta.get("max_steps", 50))
    task_type = meta.get("task_type", "unknown")
    num_episodes = int(meta.get("num_episodes", DEFAULT_NUM_EPISODES))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    lamer = LaMerMixin(num_episodes=num_episodes, traj_gamma=traj_gamma)
    trajectories: list[Trajectory] = []
    initial_obs_text: str | None = None
    total_turns = 0
    env_init_s = 0.0
    env_step_s = 0.0

    for ep_idx in range(num_episodes):
        if not lamer.should_continue and ep_idx > 0:
            break

        # --- PLAY PHASE ---
        t = time.perf_counter()
        session = await create_env_session(AlfWorldEnv, session_mode="ray", game_file=game_file, max_steps=max_steps)
        initial_obs, info = await session.reset()
        admissible_commands = info.get("admissible_commands", [])
        env_init_s += time.perf_counter() - t

        if initial_obs_text is None:
            initial_obs_text = initial_obs

        commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)

        system_content = SYSTEM_PROMPT
        if ep_idx > 0 and lamer.reflections:
            reflection_context = _format_reflections(lamer.reflections)
            system_content += f"\n\n## Past Attempt Reflections\n{reflection_context}\n\nNow you're on trial #{ep_idx + 1}. Use these reflections to improve your approach."

        initial_prompt = (
            f"Observation:\n{initial_obs}\n\n"
            f"Admissible commands:\n{commands_str}\n\n"
            f"Steps remaining: {max_steps}\n\n"
            f"What is your next action?"
        )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": initial_prompt},
        ]

        steps: list[Step] = []
        action_history_lines: list[str] = []
        won = False

        async with session:
            for turn in range(max_steps):
                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                except Exception as e:
                    logger.warning("alfworld_lamer task %s ep %d turn %d: LLM failed: %s", task.id, ep_idx, turn, e)
                    break

                content = resp.choices[0].message.content or ""
                action_str = parse_action(content)

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

                action_history_lines.append(f"Action {len(action_history_lines) + 1}: {action_str}")

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
                    user_msg += f"\nNote: Your last action '{action_str}' had no effect."
                user_msg += "\n\nWhat is your next action?"
                messages.append({"role": "user", "content": user_msg})

        total_turns += len(steps)
        ep_reward = 1.0 if won else 0.0
        lamer.record_episode(ep_reward, "\n".join(action_history_lines))

        trajectories.append(Trajectory(name=f"alfworld_ep{ep_idx}", steps=steps, reward=None))

        # --- REFLECT PHASE ---
        if not won and ep_idx < num_episodes - 1:
            curr_traj_str = "\n".join(action_history_lines)
            reflect_prompt = ALFWORLD_REFLECT_PROMPT.format(
                init_observation=initial_obs_text or "",
                current_trajectory=curr_traj_str,
            )
            reflect_messages = [{"role": "user", "content": reflect_prompt}]

            try:
                resp = await client.chat.completions.create(
                    model=config.model, messages=reflect_messages, **sampling, timeout=120,
                )
                reflect_content = resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning("alfworld_lamer task %s ep %d: reflect failed: %s", task.id, ep_idx, e)
                reflect_content = ""

            remark_text = parse_remark(reflect_content) if reflect_content else ""
            lamer.record_reflection(remark_text)

            reflect_step = Step(
                chat_completions=reflect_messages + [{"role": "assistant", "content": reflect_content}],
                model_response=reflect_content,
                action="reflect",
                thought=reflect_content,
            )
            trajectories.append(Trajectory(name=f"alfworld_reflect{ep_idx}", steps=[reflect_step], reward=None))

    # --- COMPUTE CROSS-EPISODE DISCOUNTED REWARDS ---
    discounted_rewards = lamer.discounted_rewards

    for traj in trajectories:
        if traj.name.startswith("alfworld_ep"):
            idx = int(traj.name[len("alfworld_ep"):])
            traj.reward = discounted_rewards[idx]
        elif traj.name.startswith("alfworld_reflect"):
            idx = int(traj.name[len("alfworld_reflect"):])
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
            "task_type": task_type,
            "num_episodes": num_episodes,
            "traj_gamma": traj_gamma,
        },
        is_correct=lamer.won_any,
    )
