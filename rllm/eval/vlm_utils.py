"""Shared VLM (Vision Language Model) utilities.

Image loading, MIME detection, and multimodal content block construction
for the OpenAI API.
"""

from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

# Default root for rllm datasets (overridden by RLLM_HOME env var)
_DATASETS_ROOT = os.path.join(os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm")), "datasets")


def _detect_mime_type(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _load_image_as_data_uri(rel_path: str) -> str:
    """Load an image file and return a base64 data URI."""
    abs_path = os.path.join(_DATASETS_ROOT, rel_path)
    with open(abs_path, "rb") as f:
        raw = f.read()
    mime = _detect_mime_type(raw)
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _image_to_data_uri(image_data) -> str:
    """Convert image data (bytes or path string) to a base64 data URI."""
    if isinstance(image_data, bytes):
        mime = _detect_mime_type(image_data)
        encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    if isinstance(image_data, str):
        return _load_image_as_data_uri(image_data)
    raise TypeError(f"Expected bytes or str, got {type(image_data).__name__}")


def _build_vlm_content(text: str, image_items: list) -> list[dict]:
    """Build OpenAI multimodal content blocks (image_url + text)."""
    content: list[dict] = []

    for img_data in image_items:
        try:
            data_uri = _image_to_data_uri(img_data)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
        except Exception as e:
            logger.warning("Failed to load image %s: %s", img_data if isinstance(img_data, str) else f"<{len(img_data)} bytes>", e)

    content.append({"type": "text", "text": text})
    return content
