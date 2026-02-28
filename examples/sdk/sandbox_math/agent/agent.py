"""Sandboxed math agent.

This module is uploaded into the sandbox and invoked by the worker server.
It receives a task dict and a config dict, makes LLM calls through the
proxy URL provided in ``config["base_url"]``, and returns a list of
Trajectory objects with rewards.

The agent contract:
    def rollout(task: dict, config: dict) -> list[Trajectory]

The config dict always contains:
    - base_url: proxied OpenAI-compatible endpoint with session metadata
    - session_uid: unique ID for trace correlation
    - model_id: model name to use
"""

from __future__ import annotations

import logging
import re

from openai import OpenAI

from rllm.types import Step, Trajectory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a math solver. Solve the given problem step by step.
Show your reasoning clearly, then provide the final answer inside
<answer>...</answer> tags.

For example: <answer> 42 </answer>"""


def _extract_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _check_countdown(task: dict, response_text: str) -> float:
    """Countdown reward: 1.0 if the extracted answer matches the target."""
    answer_str = _extract_answer(response_text)
    if answer_str is None:
        return 0.0
    target = task.get("target") or task.get("ground_truth")
    if target is None:
        return 0.0
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", answer_str):
            return 0.0
        result = eval(answer_str)  # noqa: S307
        return 1.0 if abs(float(result) - float(target)) < 1e-6 else 0.0
    except Exception:
        return 0.0


def rollout(task: dict, config: dict) -> list[Trajectory]:
    """Run a single math agent rollout.

    Args:
        task: Dataset example dict with keys like ``question``, ``target``.
        config: Execution config with ``base_url``, ``session_uid``, ``model_id``.

    Returns:
        A list containing one Trajectory with the reward set.
    """
    base_url = config["base_url"]
    model_id = config.get("model_id", "default")

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    question = task.get("question", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response_text = ""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)

    reward = _check_countdown(task, response_text)

    step = Step(
        input=messages,
        output=response_text,
        reward=reward,
        done=True,
    )
    traj = Trajectory(name="solver", steps=[step], reward=reward)

    logger.info("reward=%.1f response=%s", reward, response_text[:120])
    return [traj]
