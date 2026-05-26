"""Multi-turn WebShop agent flow.

The agent searches and clicks through the WebShop simulator by emitting one
action per turn in the source agent format.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from env_service import create_env_session
from env_service.webshop import WebShopEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_webshop_data import LAMER_WEBSHOP_CONFIG
except (ImportError, ValueError):
    from prepare_webshop_data import LAMER_WEBSHOP_CONFIG

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"```action\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_GENERIC_RE = re.compile(r"```\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def load_world_model_summary(file_path: str | Path | None) -> str | None:
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary")
        return summary if summary else None
    except Exception as e:
        logger.warning("Failed to load WebShop world model file %s: %s", path, e)
        return None


def _build_system_prompt(world_model_summary: str | None = None) -> str:
    world_model_section = ""
    if world_model_summary:
        world_model_section = f"""
## World Model Knowledge
The following is a summary of the environment based on exploration:

{world_model_summary}
"""

    return f"""You are a helpful shopping assistant navigating an e-commerce website. Your task is to find and purchase a product that matches the given instruction.

## Available Actions
You can perform two types of actions:

1. **search[query]**: Search for products using keywords
   - Example: `search[red running shoes size 10]`
   - Use this when you're on the search page and need to find products

2. **click[element]**: Click on a page element
   - Example: `click[Buy Now]`
   - Use this to:
     - Click on product links to view details
     - Select product options (size, color, etc.)
     - Navigate pages (Next, Prev, Back to Search)
     - Complete purchase (Buy Now)

{world_model_section}

## Navigation Tips
1. Start by searching for the product described in the instruction
2. Click on a product that looks promising to see its details
3. Select the required options (size, color, etc.) if specified
4. Click "Buy Now" when you've found the right product with correct options
5. Use "Back to Search" if you need to try different products

## Response Format
Think through your approach step by step, then provide your action in the following format:
<think>
Your reasoning and analysis here...
</think>
```action
search[your search query]
```
or
```action
click[element to click]
```

Always provide exactly ONE action at a time.
"""


def _extract_action_text(response: str) -> str | None:
    matches = _ACTION_RE.findall(response)
    if matches:
        return matches[-1].strip()
    matches = _GENERIC_RE.findall(response)
    if matches:
        return matches[-1].strip()
    return None


def _has_action_pattern(text: str) -> bool:
    return bool(re.search(r"(search\[[^\]]+\]|click\[[^\]]+\])", text, re.IGNORECASE))


def parse_action(response: str, has_search_bar: bool = False) -> str | None:
    """Extract a WebShop action string from model output."""
    action_text = _extract_action_text(response)
    if action_text:
        action_text = action_text.strip()
        if _has_action_pattern(action_text):
            match = re.search(r"(search\[[^\]]+\]|click\[[^\]]+\])", action_text, re.IGNORECASE)
            return match.group(1) if match else action_text
        if has_search_bar and action_text:
            return f"search[{action_text}]"
        return "click[back to search]"

    match = re.search(r"(search\[[^\]]+\]|click\[[^\]]+\])", response, re.IGNORECASE)
    if match:
        return match.group(1)
    stripped = response.strip()
    if stripped:
        if has_search_bar:
            return f"search[{stripped}]"
        return "click[back to search]"
    return None


def _strip_think_block(content: str) -> str:
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    return content


def _valid_action(action: str | None, available_actions: dict) -> bool:
    if action is None:
        return False
    action_lower = action.lower()
    if action_lower.startswith("search["):
        return bool(available_actions.get("has_search_bar", False)) and action_lower.endswith("]")
    if action_lower.startswith("click[") and action_lower.endswith("]"):
        inner = action[6:-1].strip().lower()
        clickables = [str(item).lower() for item in available_actions.get("clickables", [])]
        return inner in clickables
    return False


def _build_user_prompt(
    observation: str,
    instruction: str,
    available_actions: dict,
    turn: int,
    max_turns: int,
    *,
    use_available_actions: bool = True,
    action_is_valid: bool = True,
) -> str:
    header = "**Shopping Task:**" if turn == 0 else "**Current Page:**"
    prompt = f"{header}\n{instruction}\n\n" if turn == 0 else ""
    prompt += f"**Current Page:**\n{observation}\n"

    if use_available_actions:
        has_search = bool(available_actions.get("has_search_bar", False))
        clickables = [str(item) for item in available_actions.get("clickables", [])]
        prompt += "\n**Available Actions:**\n"
        if has_search:
            prompt += "- search[query]: You can search for products\n"
        if clickables:
            sample = clickables[:10]
            if len(clickables) > 10:
                prompt += f"- click[element]: Clickable elements include: {', '.join(sample)}... ({len(clickables)} total)\n"
            else:
                prompt += f"- click[element]: Clickable elements: {', '.join(clickables)}\n"

    if not action_is_valid:
        prompt += "\nYour last response did not contain a valid action, so a no-op was applied. Choose only from search[...] or click[...].\n"

    remaining = max(max_turns - turn, 0)
    prompt += f"\nSteps remaining: {remaining}\n\nWhat is your next action?"
    return prompt


@rllm.rollout(name="webshop")
async def webshop_flow(task: Task, config: AgentConfig) -> Episode:
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
    uid = config.session_uid or task.id

    t = time.perf_counter()
    logger.info("webshop rollout %s: env reset start session=%s max_steps=%d", uid, session_id, max_steps)
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
    observation, info = await session.reset()
    env_init_s = time.perf_counter() - t
    available_actions = info.get("available_actions", {})
    logger.info(
        "webshop rollout %s: env reset done in %.2fs has_search=%s clickables=%d",
        uid, env_init_s,
        available_actions.get("has_search_bar", False),
        len(available_actions.get("clickables", [])),
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    system_prompt = _build_system_prompt(world_model_summary)
    current_instruction = info.get("instruction", task.instruction if isinstance(task.instruction, str) else "")
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
    env_steps = 0
    won = False
    final_reward = 0.0
    task_score = 0.0
    last_action = None
    env_step_s = 0.0
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    async with session:
        for turn in range(max_turns):
            if env_steps >= max_steps:
                break

            model_messages = messages if use_accumulate_history else [messages[0], messages[-1]]
            try:
                if turn == 0:
                    logger.info("webshop rollout %s: first LLM request start", uid)
                resp = await client.chat.completions.create(
                    model=config.model, messages=model_messages, **sampling, timeout=120,
                )
                if turn == 0:
                    logger.info("webshop rollout %s: first LLM response received", uid)
            except Exception as e:
                logger.warning("webshop task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            raw_content = resp.choices[0].message.content or ""
            assistant_content = raw_content if use_accumulate_thinking else _strip_think_block(raw_content)
            action = parse_action(raw_content, has_search_bar=bool(available_actions.get("has_search_bar", False)))
            action_is_valid = _valid_action(action, available_actions)
            if action is None:
                action = "click[back to search]"
            last_action = action

            step_messages = list(model_messages) + [{"role": "assistant", "content": assistant_content}]

            t = time.perf_counter()
            result = await session.step(action)
            env_step_s += time.perf_counter() - t
            env_steps += 1

            observation = result.observation
            final_reward = result.reward
            won = result.won
            done = result.done
            task_score = float(result.info.get("task_score", final_reward))
            available_actions = result.info.get("available_actions", {})

            steps.append(
                Step(
                    chat_completions=step_messages,
                    observation=current_prompt,
                    thought=assistant_content,
                    model_response=assistant_content,
                    action=action,
                    reward=final_reward,
                    done=done,
                    metadata={
                        "instruction": current_instruction,
                        "task_score": task_score,
                        "won": won,
                        "action_is_valid": action_is_valid,
                    },
                )
            )

            if done or won or env_steps >= max_steps:
                break

            if use_accumulate_history:
                messages = list(step_messages)
            else:
                messages = [messages[0]]

            current_prompt = _build_user_prompt(
                observation, current_instruction, available_actions,
                turn=turn + 1, max_turns=max_turns,
                use_available_actions=use_available_actions,
                action_is_valid=action_is_valid,
            )
            messages.append({"role": "user", "content": current_prompt})

    return Episode(
        trajectories=[Trajectory(name="webshop", steps=steps)],
        metrics={"time/env_init_s": env_init_s, "time/env_step_s": env_step_s},
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
            "instruction": current_instruction,
        },
        is_correct=won,
    )
