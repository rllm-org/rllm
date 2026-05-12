"""Geo3K AgentFlow — VLM geometry problem solver.

A single-turn VLM agent that solves geometry problems from the Geometry3K
dataset. Uses plain OpenAI client with multimodal content blocks — works
identically for eval and training (the gateway handles trace capture).
"""

from __future__ import annotations

import base64
import logging
import re

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Task, Trajectory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a math problem solver with vision capabilities. You are given a \
geometry problem that includes a diagram.
Solve the problem step by step, showing your reasoning clearly.
Put your final answer in \\boxed{} notation.

For example: The answer is \\boxed{42}."""


@rllm.rollout
async def geo3k_flow(task: Task, config: AgentConfig) -> Episode:
    """Single-turn VLM geometry solver."""
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    question = task.instruction
    images = task.metadata.get("images") or []

    user_content = _build_vlm_content(question, images) if images else question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # top_k is not a chat.completions parameter; drop it from the rollout sampling params.
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    response_text = ""
    try:
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            **sampling,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)

    return Episode(
        task=task.metadata,
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


def _img_to_data_uri(img) -> str:
    """Normalize one image (bytes / dict / str / PIL) to a data URI string."""
    # Verl-format dataset row: ``{"bytes": <png-bytes>, "path": "..."}``.
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            img = img["bytes"]
        elif "path" in img and img["path"]:
            img = img["path"]  # treat as URL/URI below
        else:
            raise ValueError(f"image dict missing usable 'bytes' or 'path': keys={list(img.keys())}")
    if isinstance(img, bytes):
        mime = _detect_mime(img)
        return f"data:{mime};base64,{base64.b64encode(img).decode('utf-8')}"
    if isinstance(img, str):
        return img  # assume already a URI or URL
    # PIL Image — convert to bytes.
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _build_vlm_content(text: str, images: list) -> list[dict]:
    """Build OpenAI multimodal content blocks from text containing ``<image>`` markers.

    Mirrors verl's ``rl_dataset._build_messages`` (verl/utils/dataset/rl_dataset.py:_build_messages):
    splits the text on ``<image>`` / ``<video>`` markers and interleaves
    ``image_url`` blocks at each marker position. This is critical because the
    chat template expands an image content block into one image placeholder —
    if the literal ``<image>`` ALSO survives in the rendered text, vLLM emits
    a malformed prompt where one image is shown and the literal ``<image>``
    string is appended (the model then "doesn't see" the image properly).
    """
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        return [{"type": "text", "text": text}]

    segments = [s for s in re.split(r"(<image>|<video>)", text) if s != ""]
    content: list[dict] = []
    img_idx = 0
    for seg in segments:
        if seg == "<image>":
            if img_idx >= len(valid_images):
                raise ValueError(f"<image> marker count exceeds image count ({len(valid_images)}) in {text!r}")
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_uri(valid_images[img_idx])}})
            img_idx += 1
        elif seg == "<video>":
            # Geo3K is image-only; surface unexpected video markers.
            raise NotImplementedError("<video> markers not supported in geo3k_flow")
        else:
            content.append({"type": "text", "text": seg})

    # If the source text didn't include ``<image>`` markers (some datasets
    # omit them), prepend any unconsumed images so the model still sees them.
    while img_idx < len(valid_images):
        content.insert(0, {"type": "image_url", "image_url": {"url": _img_to_data_uri(valid_images[img_idx])}})
        img_idx += 1

    return content
