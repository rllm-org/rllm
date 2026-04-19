from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


def normalize_tools(tools: list) -> list[dict]:
    """Normalize tools to OpenAI-schema dicts for apply_chat_template.

    Accepts rllm Tool instances, dicts, JSON strings, or any pydantic model
    exposing model_dump().
    """
    from rllm.tools.tool_base import Tool

    out: list[dict] = []
    for tool in tools:
        if isinstance(tool, Tool):
            out.append(tool.json)
        elif isinstance(tool, dict):
            out.append(tool)
        elif isinstance(tool, str):
            out.append(json.loads(tool))
        elif hasattr(tool, "model_dump"):
            out.append(tool.model_dump(exclude_none=True))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")
    return out


def _image_part_to_value(part: dict) -> Any:
    """Return the image payload from an OpenAI-style image content part.

    Accepts OpenAI's ``{"type": "image_url", "image_url": {"url": ...}}``
    and the simpler ``{"type": "image", "image": ...}`` shape. The payload
    may be a URL string, data-URI, base64 string, or PIL.Image.Image.
    """
    if part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    if part.get("type") == "image":
        return part.get("image")
    return None


def normalize_messages_for_images(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI-style image content parts to the ``message["images"]``
    side-channel convention used internally by ``QwenChatTemplateParser`` and
    by the verl rollout engine's image path.

    For each message whose ``content`` is a list of parts, image parts are
    collected into ``message["images"]`` (a list of URL strings, data URIs,
    or PIL images) and text parts are concatenated into a single
    ``content: str``. If ``message["images"]`` is already populated the
    message is left alone — this makes the function idempotent and lets
    callers mix the two wire conventions.

    Other fields (role, tool_calls, reasoning, etc.) are preserved.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue

        if msg.get("images"):
            # Already normalized; keep as-is but collapse list content to text
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            new_msg = dict(msg)
            new_msg["content"] = "".join(text_parts)
            out.append(new_msg)
            continue

        text_parts: list[str] = []
        images: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype in ("image", "image_url"):
                value = _image_part_to_value(part)
                if value is not None:
                    images.append(value)

        new_msg = dict(msg)
        new_msg["content"] = "".join(text_parts)
        if images:
            new_msg["images"] = images
        out.append(new_msg)
    return out


def normalize_messages_for_tinker(messages: list[dict]) -> list[dict]:
    """Normalize messages into a form tinker-cookbook's openai_compat expects,
    preserving image parts.

    Accepts either the OpenAI content-parts wire format (``content: list[dict]``
    with ``{"type": "image_url", ...}`` parts) or the side-channel form
    (``message["images"]`` with text-only ``content: str``). Returns messages
    whose ``content`` is a list of ``{"type": "text", "text": ...}`` and
    ``{"type": "image", "image": ...}`` parts — the shape tinker-cookbook's
    ContentPart list accepts.

    If a message has neither images nor list content it is passed through
    unchanged so text-only paths remain bit-identical.
    """
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        has_images = bool(msg.get("images"))
        is_list_content = isinstance(content, list)

        if not has_images and not is_list_content:
            out.append(msg)
            continue

        parts: list[dict] = []

        if is_list_content:
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text", "")
                    if text:
                        parts.append({"type": "text", "text": text})
                elif ptype in ("image", "image_url"):
                    value = _image_part_to_value(part)
                    if value is not None:
                        parts.append({"type": "image", "image": value})
        elif isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})

        if has_images and not is_list_content:
            for image in msg["images"]:
                if isinstance(image, dict) and "image" in image:
                    parts.append({"type": "image", "image": image["image"]})
                else:
                    parts.append({"type": "image", "image": image})

        new_msg = dict(msg)
        new_msg["content"] = parts
        new_msg.pop("images", None)
        out.append(new_msg)
    return out


def extract_images_pil(messages: list[dict], processor) -> list[Image.Image]:
    """Resolve all image payloads in ``message["images"]`` to PIL images.

    Uses ``qwen_vl_utils.fetch_image`` for URL / data-URI / path / PIL resolution,
    sized against the multimodal ``processor.image_processor.patch_size``.
    Expects messages to have already been normalized via
    ``normalize_messages_for_images`` (so images live in the side channel).
    """
    from qwen_vl_utils import fetch_image

    patch_size = processor.image_processor.patch_size
    out: list[Image.Image] = []
    for msg in messages:
        images = msg.get("images")
        if not images:
            continue
        for image in images:
            image_dict = image if isinstance(image, dict) else {"image": image}
            out.append(fetch_image(image_dict, image_patch_size=patch_size))
    return out
