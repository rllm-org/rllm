"""VLM TITO (Token-In-Token-Out) extensions for vLLM multimodal inference.

Independent from Responses API translation (see _responses_compat.py):
this module handles the vLLM-side concerns of dedup + multi_modal_data.

Background:
  HF processors expand a single <image> reference into N consecutive pad
  tokens (e.g. Qwen2.5-VL: 64x 151655) so the text tokenizer produces a
  slot for every vision patch. vLLM's /v1/completions endpoint, when it
  receives prompt_token_ids + multi_modal_data, expects the OPPOSITE: one
  placeholder per image. Feeding N-run prompt_ids without dedup either
  errors out or (worse) silently skips the vision encoder — the model
  effectively becomes text-only despite receiving pixels.

  apply_vlm_tito is the single call site that reconciles the two views:
  it compresses each pad-run into one placeholder and injects a
  multi_modal_data dict built from the caller-supplied data URLs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ImagePosition:
    """Position of a single deduped image placeholder in the token sequence."""

    index: int  # position in deduped token sequence
    original_count: int  # how many pad tokens were compressed


def resolve_image_pad_token_id(renderer) -> int | None:
    """Best-effort extraction of the image-pad token id from a renderer.

    Lookup order:
      1. renderer.image_token_id — direct attribute (would be ideal, but the
         renderers 0.x library doesn't ship this — see T2 verification 2026-07-16)
      2. renderer.mm_token_type_id_map — Qwen3.5+ exposes ``{token_id: type_id}``
         where type_id 1 = image, 2 = video (checked on Qwen35Renderer,
         instance-level property returning e.g. {151655: 1, 151656: 2})
      3. None — caller should fall back to ``RLLM_VLM_PAD_TOKEN_ID`` env

    Returns None if no source resolves — apply_vlm_tito will then consult the
    env fallback, or (in strict mode) raise.
    """
    direct = getattr(renderer, "image_token_id", None)
    if isinstance(direct, int):
        return direct
    mm_map = getattr(renderer, "mm_token_type_id_map", None)
    if isinstance(mm_map, dict):
        # Qwen VL family convention: type_id == 1 marks the image pad token.
        for token_id, type_id in mm_map.items():
            if type_id == 1 and isinstance(token_id, int):
                return token_id
    return None


def dedup_media_tokens(
    token_ids: list[int],
    pad_token_id: int,
) -> tuple[list[int], list[ImagePosition]]:
    """Compress consecutive pad tokens into a single placeholder.

    HF processor expands <image> into N consecutive pad tokens (e.g. Qwen2.5-VL:
    64x 151655), but vLLM /v1/completions expects ONE placeholder per image plus
    multi_modal_data. Runs of non-pad tokens pass through unchanged.

    Example:
        dedup_media_tokens([1, 2, PAD, PAD, PAD, 3, PAD, PAD, 4], PAD)
        -> ([1, 2, PAD, 3, PAD, 4],
            [ImagePosition(index=2, original_count=3),
             ImagePosition(index=4, original_count=2)])
    """
    deduped: list[int] = []
    positions: list[ImagePosition] = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == pad_token_id:
            run_start = i
            while i < len(token_ids) and token_ids[i] == pad_token_id:
                i += 1
            positions.append(
                ImagePosition(index=len(deduped), original_count=i - run_start)
            )
            deduped.append(pad_token_id)
        else:
            deduped.append(token_ids[i])
            i += 1
    return deduped, positions


@dataclass
class VLMImageChannel:
    """Collect data URLs for vLLM ``multi_modal_data``.

    vLLM's HTTP API (``/v1/completions``) accepts ``multi_modal_data`` with
    image content as a list of URL strings (http(s):// or data:...;base64).
    We keep the URLs as strings — no PIL decode/re-encode round trip — so
    (a) pillow is not a required dep, and (b) the payload stays JSON-serializable
    for the httpx POST.
    """

    urls: list[str] = field(default_factory=list)

    def add_from_data_url(self, data_url: str) -> None:
        """Accept a data URL (``data:...;base64,<payload>``) or http(s) URL.

        Raises ValueError if the argument is not a well-formed string URL.
        """
        if not isinstance(data_url, str) or not data_url:
            raise ValueError(f"Expected image URL string, got: {type(data_url).__name__}={data_url!r}")
        if not (data_url.startswith("data:") or data_url.startswith("http://") or data_url.startswith("https://")):
            raise ValueError(f"Expected data:...;base64 or http(s) URL, got: {data_url[:40]!r}")
        self.urls.append(data_url)

    def build_multi_modal_data(self) -> dict:
        return {"image": list(self.urls)}


def apply_vlm_tito(
    completions_body: dict,
    images: list[str],
    pad_token_id: int | None = None,
) -> None:
    """Dedup pad tokens + inject multi_modal_data. Mutates completions_body in place.

    No-op if:
      - images is empty
      - pad_token_id cannot be resolved (arg None + no RLLM_VLM_PAD_TOKEN_ID env)
      - dedup produces a pad-run count != image count

    In strict mode (RLLM_STRICT_VLM=1): raises RuntimeError on the last two
    conditions instead of silently skipping. Training should always run strict
    so a broken vision path produces a crash, not silently text-only rollouts.
    """
    if not images:
        return
    strict = os.environ.get("RLLM_STRICT_VLM", "0") == "1"

    if pad_token_id is None:
        env_val = os.environ.get("RLLM_VLM_PAD_TOKEN_ID")
        if env_val:
            pad_token_id = int(env_val)
        else:
            msg = "VLM TITO: pad_token_id not provided and RLLM_VLM_PAD_TOKEN_ID unset"
            if strict:
                raise RuntimeError(msg)
            logger.warning("%s. Skipping (set RLLM_STRICT_VLM=1 to raise).", msg)
            return

    channel = VLMImageChannel()
    for url in images:
        channel.add_from_data_url(url)

    prompt = completions_body.get("prompt")
    if not isinstance(prompt, list):
        msg = f"VLM TITO: completions_body['prompt'] must be a list of token IDs, got {type(prompt).__name__}"
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s. Skipping.", msg)
        return

    deduped, positions = dedup_media_tokens(prompt, pad_token_id)
    if len(positions) != len(channel.urls):
        msg = (
            f"VLM TITO mismatch: {len(positions)} pad runs vs "
            f"{len(channel.urls)} images"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s. Skipping (set RLLM_STRICT_VLM=1 to raise).", msg)
        return

    completions_body["prompt"] = deduped
    completions_body["multi_modal_data"] = channel.build_multi_modal_data()
