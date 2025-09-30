"""Multimodal data utilities for RLLM."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# Import Verl's vision utilities - these are the core multimodal processing functions
try:
    from verl.utils.dataset.vision_utils import process_image, process_video
    from verl.workers.rollout.schemas import AsyncRolloutRequest, Message
    VERL_MULTIMODAL_AVAILABLE = True
except ImportError:
    VERL_MULTIMODAL_AVAILABLE = False
    # Fallback stubs for type hints
    class AsyncRolloutRequest:
        pass
    class Message:
        pass

    def process_image(image):  # type: ignore[override]
        return image

    def process_video(video):  # type: ignore[override]
        return video


DATA_URI_PREFIX = "data:image/"


def as_pil_image(image: Any) -> Image.Image | None:
    """Convert supported payloads (PIL image, data URI, byte dict) to a PIL image."""

    if hasattr(image, "mode") and hasattr(image, "size"):
        return image  # Already a PIL image

    if isinstance(image, str) and image.startswith(DATA_URI_PREFIX):
        try:
            header, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:
                return None

        # HuggingFace datasets often store images as dicts with "path" pointing to a
        # file or a data URI. Handle both cases, falling back to disk loading.
        data_str: str | None = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]

        if data_str:
            if data_str.startswith(DATA_URI_PREFIX):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    return None
            else:
                try:
                    return Image.open(data_str).convert("RGB")
                except Exception:
                    return None

    return None


def ensure_multimodal_available():
    """Ensure Verl multimodal capabilities are available."""
    if not VERL_MULTIMODAL_AVAILABLE:
        raise ImportError(
            "Verl multimodal support not available. Please ensure Verl is properly installed."
        )


@dataclass
class MultimodalMessage:
    """A message that can contain text, images, and videos.

    This is a lightweight wrapper around Verl's Message format for easier use in RLLM.
    """

    role: str  # "user", "assistant", or "system"
    text: Optional[str] = None
    images: Optional[List[Union[str, Dict, Image.Image]]] = None
    videos: Optional[List[Dict]] = None

    def to_verl_message(self) -> Dict[str, Any]:
        """Convert to Verl's Message format."""
        ensure_multimodal_available()

        content = []

        # Add text content
        if self.text:
            content.append({"type": "text", "text": self.text})

        # Add image content - process through Verl's utilities
        if self.images:
            processed_images = []
            for image in self.images:
                pil_image = as_pil_image(image)
                if pil_image is not None:
                    processed_images.append(pil_image.convert("RGB"))
                else:
                    processed_images.append(process_image(image))
            content.append({"type": "image", "image": processed_images})

        # Add video content - process through Verl's utilities
        if self.videos:
            processed_videos = []
            for video in self.videos:
                processed_video = process_video(video)
                processed_videos.append(processed_video)
            content.append({"type": "video", "video": processed_videos})

        return {
            "role": self.role,
            "content": content if content else self.text or ""
        }


def create_multimodal_conversation(messages: List[MultimodalMessage]) -> List[Dict[str, Any]]:
    """Create a conversation from MultimodalMessage objects.

    Args:
        messages: List of MultimodalMessage objects

    Returns:
        List of messages in Verl format, ready for training
    """
    return [msg.to_verl_message() for msg in messages]


def extract_multimodal_data(messages: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Extract multimodal data from messages for Verl processing.

    This function extracts all images and videos from a conversation,
    which is the format expected by Verl's AsyncRolloutRequest.

    Args:
        messages: List of messages in Verl format

    Returns:
        Dictionary with 'image' and 'video' keys containing processed media
    """
    ensure_multimodal_available()

    all_images = []
    all_videos = []

    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue

        for item in content:
            if item.get("type") == "image":
                images = item.get("image", [])
                if isinstance(images, list):
                    all_images.extend(images)
                else:
                    all_images.append(images)
            elif item.get("type") == "video":
                videos = item.get("video", [])
                if isinstance(videos, list):
                    all_videos.extend(videos)
                else:
                    all_videos.append(videos)

    return {
        "image": all_images,
        "video": all_videos
    }


# Convenience functions for common use cases

def create_text_message(role: str, text: str) -> MultimodalMessage:
    """Create a text-only message."""
    return MultimodalMessage(role=role, text=text)


def create_image_message(
    role: str,
    text: str,
    images: List[Union[str, Dict, Image.Image]]
) -> MultimodalMessage:
    """Create a message with text and images."""
    return MultimodalMessage(role=role, text=text, images=images)


def create_video_message(
    role: str,
    text: str,
    videos: List[Dict]
) -> MultimodalMessage:
    """Create a message with text and videos."""
    return MultimodalMessage(role=role, text=text, videos=videos)


def create_multimodal_message(
    role: str,
    text: str,
    images: Optional[List[Union[str, Dict, Image.Image]]] = None,
    videos: Optional[List[Dict]] = None
) -> MultimodalMessage:
    """Create a message with text, images, and/or videos."""
    return MultimodalMessage(role=role, text=text, images=images, videos=videos)
