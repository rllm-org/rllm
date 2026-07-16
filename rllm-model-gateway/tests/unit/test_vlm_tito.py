"""Unit tests for vlm_tito.py — dedup_media_tokens, VLMImageChannel, apply_vlm_tito.

Phase D unit-level coverage. No network / no upstream; pure function behavior.
"""

import logging
from unittest.mock import patch

import pytest
from rllm_model_gateway.vlm_tito import (
    ImagePosition,
    VLMImageChannel,
    apply_vlm_tito,
    dedup_media_tokens,
    resolve_image_pad_token_id,
)

PAD = 151655  # Qwen2.5-VL <|image_pad|> id — arbitrary here, just needs to be consistent
_URL_A = "data:image/png;base64,IMG_A"
_URL_B = "data:image/png;base64,IMG_B"


# ---------------------------------------------------------------------------
# dedup_media_tokens
# ---------------------------------------------------------------------------


class TestDedupMediaTokens:
    def test_basic_single_image(self):
        deduped, positions = dedup_media_tokens([1, 2, PAD, PAD, PAD, 3], PAD)
        assert deduped == [1, 2, PAD, 3]
        assert positions == [ImagePosition(index=2, original_count=3)]

    def test_two_pad_runs(self):
        deduped, positions = dedup_media_tokens([1, PAD, PAD, 2, PAD, PAD, 3], PAD)
        assert deduped == [1, PAD, 2, PAD, 3]
        assert positions == [
            ImagePosition(index=1, original_count=2),
            ImagePosition(index=3, original_count=2),
        ]

    def test_no_pads_is_identity(self):
        deduped, positions = dedup_media_tokens([1, 2, 3, 4], PAD)
        assert deduped == [1, 2, 3, 4]
        assert positions == []

    def test_all_pads(self):
        deduped, positions = dedup_media_tokens([PAD, PAD, PAD], PAD)
        assert deduped == [PAD]
        assert positions == [ImagePosition(index=0, original_count=3)]

    def test_pad_at_boundaries(self):
        deduped, positions = dedup_media_tokens([PAD, 1, 2, PAD], PAD)
        assert deduped == [PAD, 1, 2, PAD]
        assert len(positions) == 2

    def test_single_pad_between_content(self):
        # A single pad token is a valid deduped placeholder (original_count=1).
        deduped, positions = dedup_media_tokens([1, PAD, 2], PAD)
        assert deduped == [1, PAD, 2]
        assert positions == [ImagePosition(index=1, original_count=1)]

    def test_empty_input(self):
        deduped, positions = dedup_media_tokens([], PAD)
        assert deduped == []
        assert positions == []


# ---------------------------------------------------------------------------
# VLMImageChannel
# ---------------------------------------------------------------------------


class TestVLMImageChannel:
    def test_add_data_url_preserves_string(self):
        ch = VLMImageChannel()
        ch.add_from_data_url(_URL_A)
        assert ch.urls == [_URL_A]

    def test_add_multiple_preserves_order(self):
        ch = VLMImageChannel()
        ch.add_from_data_url(_URL_A)
        ch.add_from_data_url(_URL_B)
        assert ch.urls == [_URL_A, _URL_B]

    def test_add_http_url_ok(self):
        ch = VLMImageChannel()
        ch.add_from_data_url("https://example.com/img.png")
        assert ch.urls == ["https://example.com/img.png"]

    def test_add_rejects_non_string(self):
        ch = VLMImageChannel()
        with pytest.raises(ValueError):
            ch.add_from_data_url(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ch.add_from_data_url(b"data:...")  # type: ignore[arg-type]

    def test_add_rejects_malformed_scheme(self):
        ch = VLMImageChannel()
        with pytest.raises(ValueError):
            ch.add_from_data_url("not a url at all")
        with pytest.raises(ValueError):
            ch.add_from_data_url("ftp://something")

    def test_build_multi_modal_data(self):
        ch = VLMImageChannel()
        ch.add_from_data_url(_URL_A)
        ch.add_from_data_url(_URL_B)
        mmd = ch.build_multi_modal_data()
        assert mmd == {"image": [_URL_A, _URL_B]}


# ---------------------------------------------------------------------------
# apply_vlm_tito
# ---------------------------------------------------------------------------


class TestApplyVLMTITO:
    def test_noop_when_images_empty(self):
        body = {"prompt": [1, 2, 3]}
        apply_vlm_tito(body, images=[], pad_token_id=PAD)
        assert body == {"prompt": [1, 2, 3]}
        assert "multi_modal_data" not in body

    def test_skip_when_pad_id_absent_lenient(self):
        # RLLM_STRICT_VLM unset -> warning + skip
        body = {"prompt": [1, PAD, PAD, 2]}
        with patch.dict("os.environ", {}, clear=False) as _:
            # Ensure the flag isn't inherited from the outer env
            import os as _os

            _os.environ.pop("RLLM_STRICT_VLM", None)
            _os.environ.pop("RLLM_VLM_PAD_TOKEN_ID", None)
            apply_vlm_tito(body, images=[_URL_A], pad_token_id=None)
        assert body == {"prompt": [1, PAD, PAD, 2]}
        assert "multi_modal_data" not in body

    def test_strict_raises_when_pad_id_absent(self):
        body = {"prompt": [1, PAD, PAD, 2]}
        with patch.dict("os.environ", {"RLLM_STRICT_VLM": "1"}, clear=False):
            import os as _os

            _os.environ.pop("RLLM_VLM_PAD_TOKEN_ID", None)
            with pytest.raises(RuntimeError, match="pad_token_id not provided"):
                apply_vlm_tito(body, images=[_URL_A], pad_token_id=None)

    def test_env_fallback_provides_pad_id(self):
        body = {"prompt": [1, PAD, PAD, 2]}
        with patch.dict("os.environ", {"RLLM_VLM_PAD_TOKEN_ID": str(PAD)}, clear=False):
            apply_vlm_tito(body, images=[_URL_A], pad_token_id=None)
        assert body["prompt"] == [1, PAD, 2]
        assert body["multi_modal_data"] == {"image": [_URL_A]}

    def test_full_success_path(self):
        body = {"prompt": [1, PAD, PAD, PAD, 2, PAD, PAD, 3]}
        apply_vlm_tito(body, images=[_URL_A, _URL_B], pad_token_id=PAD)
        assert body["prompt"] == [1, PAD, 2, PAD, 3]
        assert body["multi_modal_data"] == {"image": [_URL_A, _URL_B]}

    def test_mismatch_lenient_skips(self, caplog):
        # 2 pad runs vs 1 image → mismatch → warn + skip
        body = {"prompt": [1, PAD, 2, PAD, 3]}
        import os as _os

        _os.environ.pop("RLLM_STRICT_VLM", None)
        with caplog.at_level(logging.WARNING, logger="rllm_model_gateway.vlm_tito"):
            apply_vlm_tito(body, images=[_URL_A], pad_token_id=PAD)
        assert body == {"prompt": [1, PAD, 2, PAD, 3]}
        assert "multi_modal_data" not in body
        assert any("mismatch" in r.message for r in caplog.records)

    def test_mismatch_strict_raises(self):
        body = {"prompt": [1, PAD, 2, PAD, 3]}
        with patch.dict("os.environ", {"RLLM_STRICT_VLM": "1"}, clear=False):
            with pytest.raises(RuntimeError, match="mismatch"):
                apply_vlm_tito(body, images=[_URL_A], pad_token_id=PAD)

    def test_prompt_not_list_skipped_lenient(self):
        body = {"prompt": "a string"}
        import os as _os

        _os.environ.pop("RLLM_STRICT_VLM", None)
        apply_vlm_tito(body, images=[_URL_A], pad_token_id=PAD)
        assert body == {"prompt": "a string"}

    def test_prompt_not_list_strict_raises(self):
        body = {"prompt": "a string"}
        with patch.dict("os.environ", {"RLLM_STRICT_VLM": "1"}, clear=False):
            with pytest.raises(RuntimeError, match="must be a list"):
                apply_vlm_tito(body, images=[_URL_A], pad_token_id=PAD)


# ---------------------------------------------------------------------------
# resolve_image_pad_token_id — renderer introspection helper
# ---------------------------------------------------------------------------


class _RendererWithAttr:
    """Ideal case: renderer exposes image_token_id directly (T2 aspirational)."""
    image_token_id = 999


class _RendererWithMap:
    """Real case (Qwen35Renderer 2026-07): only mm_token_type_id_map."""
    # {token_id: type_id}, type_id 1 = image, 2 = video (Qwen VL convention)
    mm_token_type_id_map = {151655: 1, 151656: 2}


class _RendererWithMapNoImage:
    """Edge: mm_map but no type_id 1 entry (unlikely, but be defensive)."""
    mm_token_type_id_map = {151656: 2}


class _RendererBare:
    """Text-only renderer — no image machinery."""
    pass


class TestResolveImagePadTokenId:
    def test_direct_attribute_takes_precedence(self):
        assert resolve_image_pad_token_id(_RendererWithAttr()) == 999

    def test_falls_back_to_mm_map(self):
        # Qwen35Renderer real behavior: T2 finding from 2026-07-16
        assert resolve_image_pad_token_id(_RendererWithMap()) == 151655

    def test_mm_map_without_image_type_returns_none(self):
        assert resolve_image_pad_token_id(_RendererWithMapNoImage()) is None

    def test_bare_renderer_returns_none(self):
        assert resolve_image_pad_token_id(_RendererBare()) is None

    def test_none_renderer_returns_none(self):
        # Defensive — proxy guards against None renderer explicitly, but the
        # helper should still behave sanely on any junk input.
        assert resolve_image_pad_token_id(None) is None
