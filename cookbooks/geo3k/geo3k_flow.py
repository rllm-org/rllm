"""Geo3K AgentFlow — VLM geometry problem solver.

A single-turn VLM agent that solves geometry problems from the Geometry3K
dataset. Uses plain OpenAI client with multimodal content blocks — works
identically for eval and training (the gateway handles trace capture).
"""

from __future__ import annotations

import base64
import logging

from openai import OpenAI

import rllm
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Trajectory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a math problem solver with vision capabilities. You are given a \
geometry problem that includes a diagram.
Solve the problem step by step, showing your reasoning clearly.
Put your final answer in \\boxed{} notation.

For example: The answer is \\boxed{42}."""


@rllm.rollout
def geo3k_flow(task: Task, config: AgentConfig) -> Episode:
    """Single-turn VLM geometry solver."""
    data = task.data
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    question = data.get("question", "")
    images = data.get("images", [])

    user_content = _build_vlm_content(question, images) if images else question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response_text = ""
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.6,
            max_tokens=2048,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)

    return Episode(
        task=data,
        trajectories=[Trajectory(name="solver", steps=[])],
        artifacts={"answer": response_text},
    )


def _detect_mime(data: bytes) -> str:
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _build_vlm_content(text: str, images: list) -> list[dict]:
    """Build OpenAI multimodal content blocks from text + image data."""
    content: list[dict] = []
    for img in images:
        if img is None:
            continue
        if isinstance(img, bytes):
            mime = _detect_mime(img)
            encoded = base64.b64encode(img).decode("utf-8")
            data_uri = f"data:{mime};base64,{encoded}"
        elif isinstance(img, str):
            data_uri = img  # assume already a URI or URL
        else:
            # PIL Image — convert to bytes
            import io

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{encoded}"
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    content.append({"type": "text", "text": text})
    return content
