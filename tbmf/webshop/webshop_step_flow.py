"""Non-cumulative (step-based) WebShop agent flow.

Each turn constructs a fresh, self-contained prompt using the LaMer
prompt template with a sliding window of recent action/observation history.
This produces independent steps that the verl transform handles as
separate training rows -- enabling stepwise GRPO with advantage broadcasting.
"""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from .prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from .webshop_env import WebShopEnv
    from .webshop_flow import parse_action, _run_env_reset, _run_env_io
    from .webshop_prompt import get_webshop_prompt
except ImportError:  # pragma: no cover - script fallback
    from prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from webshop_env import WebShopEnv
    from webshop_flow import parse_action, _run_env_reset, _run_env_io
    from webshop_prompt import get_webshop_prompt

logger = logging.getLogger(__name__)

_HISTORY_LENGTH = 5


def _format_history(history: list[dict], history_length: int = _HISTORY_LENGTH) -> str:
    """Format action/obs history with a sliding window of recent entries.

    Matches verl-agent SimpleMemory.fetch() behavior for WebShop:
    only the last ``history_length`` entries are included.
    """
    if not history:
        return ""
    recent = history[-history_length:]
    start_idx = len(history) - len(recent)
    lines = []
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(f"Action {step_num}: {rec['action']}\nObservation {step_num}: {rec['obs']}")
    return "\n".join(lines)


def _format_available_actions(available_actions: dict) -> str:
    """Format available actions for the prompt (matching verl-agent format)."""
    lines = []
    if available_actions.get("has_search_bar", False):
        lines.append("'search[<your query>]'")
    for item in available_actions.get("clickables", []):
        item_str = str(item).lower()
        if item_str != "search":
            lines.append(f"'click[{item_str}]'")
    return ",\n".join(lines)


def _extract_task(observation: str) -> str:
    """Extract task description from initial observation.

    The observation is typically formatted as:
        WebShop [SEP] Instruction: [SEP] <task description> [SEP] ...
    """
    parts = observation.split(" [SEP] ")
    for i, part in enumerate(parts):
        if "instruction:" in part.lower():
            if i + 1 < len(parts):
                return parts[i + 1].strip()
    return ""


def _format_obs(observation: str, task_description: str) -> str:
    """Strip the instruction prefix from observation text.

    The raw observation contains the task instruction repeated; we strip
    everything up to and including that to keep only the page content.
    """
    # Find the task description in the observation and strip everything before it
    parts = observation.split(" [SEP] ")
    # Skip WebShop header, Instruction marker, and the task itself
    # Keep remaining parts as the observation
    content_parts = []
    found_task = False
    for part in parts:
        if found_task:
            content_parts.append(part)
        elif part.strip() == task_description.strip():
            found_task = True
    if content_parts:
        return " [SEP] ".join(content_parts).strip()
    # Fallback: return the full observation
    return observation


@rllm.rollout(name="webshop_step")
async def webshop_step_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive a WebShop environment with step-independent LLM calls."""
    meta = task.metadata or {}
    session_id = int(meta.get("session_id"))
    max_steps = int(meta.get("max_steps", LAMER_WEBSHOP_CONFIG["max_steps"]))
    max_turns = int(meta.get("max_turns", max_steps))
    seed = int(meta.get("seed", LAMER_WEBSHOP_CONFIG["env_seed"]))
    observation_mode = str(meta.get("observation_mode", LAMER_WEBSHOP_CONFIG["observation_mode"]))
    num_products = meta.get("num_products", LAMER_WEBSHOP_CONFIG["num_products"])
    human_goals = bool(meta.get("human_goals", LAMER_WEBSHOP_CONFIG["human_goals"]))
    file_path = meta.get("file_path")
    attr_path = meta.get("attr_path")
    uid = config.session_uid or task.id

    env = WebShopEnv(
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
    logger.info("webshop_step rollout %s: env reset start session=%s max_steps=%d", uid, session_id, max_steps)
    observation, info = await _run_env_reset(env.reset, seed)
    env_init_s = time.perf_counter() - t
    available_actions = info.get("available_actions", {})
    logger.info(
        "webshop_step rollout %s: env reset done in %.2fs has_search=%s clickables=%d",
        uid,
        env_init_s,
        available_actions.get("has_search_bar", False),
        len(available_actions.get("clickables", [])),
    )

    # Extract task description from the initial observation or info
    task_description = info.get("instruction", "")
    if not task_description:
        task_description = _extract_task(observation)
    if not task_description:
        task_description = task.instruction if isinstance(task.instruction, str) else ""

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    steps: list[Step] = []
    history: list[dict] = []
    won = False
    done = False
    env_steps = 0
    last_action: str | None = None
    env_step_s = 0.0
    final_reward = 0.0
    task_score = 0.0

    try:
        for turn in range(max_turns):
            if done or won or env_steps >= max_steps:
                break

            # Format observation for history (strip instruction prefix)
            obs_text = _format_obs(observation, task_description) if turn == 0 else observation

            # Build fresh prompt each turn
            curr_traj = _format_history(history, history_length=_HISTORY_LENGTH)
            avail_str = _format_available_actions(available_actions)
            prompt = get_webshop_prompt(
                phase="play",
                turn_idx=turn,
                traj_idx=0,
                task_description=task_description,
                curr_traj=curr_traj,
                past_traj={},
                admissible_actions=avail_str,
                reflection="",
            )

            messages = [{"role": "user", "content": prompt}]

            try:
                if turn == 0:
                    logger.info("webshop_step rollout %s: first LLM request start", uid)
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
                if turn == 0:
                    logger.info("webshop_step rollout %s: first LLM response received", uid)
            except Exception as e:
                logger.warning("webshop_step task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            action = parse_action(content, has_search_bar=bool(available_actions.get("has_search_bar", False)))
            if action is None:
                action = "click[back to search]"
            last_action = action

            # Step messages for this turn (non-cumulative)
            step_messages = messages + [{"role": "assistant", "content": content}]

            # Execute action in environment
            t = time.perf_counter()
            observation, reward, done, info = await _run_env_io(env.step, action)
            env_step_s += time.perf_counter() - t
            env_steps += 1
            final_reward = float(reward)
            task_score = float(info.get("task_score", final_reward))
            won = bool(info.get("won", False))
            available_actions = info.get("available_actions", {})

            # Append to history for sliding window
            history.append({"action": action, "obs": observation})

            steps.append(
                Step(
                    chat_completions=step_messages,
                    observation=prompt,
                    model_response=content,
                    action=action,
                    thought=content,
                    reward=float(reward),
                    done=bool(done),
                    metadata={
                        "instruction": task_description,
                        "task_score": task_score,
                        "won": won,
                    },
                )
            )

            if done or won or env_steps >= max_steps:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            await _run_env_io(close)

    return Episode(
        trajectories=[Trajectory(name="webshop_step", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "success": won,
            "task_score": task_score,
            "reward": final_reward,
            "turns": len(steps),
            "env_steps": env_steps,
            "last_action": last_action,
            "session_id": session_id,
            "max_steps": max_steps,
            "max_turns": max_turns,
            "observation_mode": observation_mode,
            "num_products": num_products,
            "human_goals": human_goals,
            "instruction": task_description,
        },
        is_correct=won,
    )
