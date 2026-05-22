from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}]},
    # {"role": "tool", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {"role": "assistant", "content": "Let me search for Java information.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}]},
]


def normalize_tools(tools: list) -> list[dict]:
    """Normalize tools to OpenAI-schema dicts for apply_chat_template."""
    from rllm.tools.tool_base import Tool

    normalized = []
    for tool in tools:
        if isinstance(tool, Tool):
            normalized.append(tool.json)
        elif isinstance(tool, dict):
            normalized.append(tool)
        elif isinstance(tool, str):
            normalized.append(json.loads(tool))
        elif hasattr(tool, "model_dump"):
            normalized.append(tool.model_dump(exclude_none=True))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")
    return normalized


def _image_part_to_value(part: dict) -> Any:
    if part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    if part.get("type") == "image":
        return part.get("image")
    return None


def normalize_messages_for_images(messages: list[dict]) -> list[dict]:
    """Move OpenAI image content parts into the message['images'] side channel."""
    normalized = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        if message.get("images"):
            normalized_message = dict(message)
            normalized_message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
            normalized.append(normalized_message)
            continue

        text_parts = []
        images = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") in ("image", "image_url"):
                value = _image_part_to_value(part)
                if value is not None:
                    images.append(value)

        normalized_message = dict(message)
        normalized_message["content"] = "".join(text_parts)
        if images:
            normalized_message["images"] = images
        normalized.append(normalized_message)
    return normalized


def extract_images_pil(messages: list[dict], processor) -> list[Image.Image]:
    """Resolve normalized message['images'] payloads to PIL images."""
    from qwen_vl_utils import fetch_image

    patch_size = processor.image_processor.patch_size
    images = []
    for message in messages:
        for image in message.get("images") or []:
            image_dict = image if isinstance(image, dict) else {"image": image}
            images.append(fetch_image(image_dict, image_patch_size=patch_size))
    return images
