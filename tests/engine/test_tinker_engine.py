"""Tests for tinker_engine utility functions and TinkerEngine static helpers."""

from unittest.mock import MagicMock

import tinker
from tinker.types import ImageChunk

from rllm.engine.rollout.tinker_engine import (
    _flat_token_input_length,
    _flat_token_input_to_model_input,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_image_chunk(length: int = 16):
    """Create a real tinker ImageChunk with the given token length."""
    return ImageChunk(data=b"\x89PNG", format="png", expected_tokens=length)


# ------------------------------------------------------------------
# _flat_token_input_to_model_input
# ------------------------------------------------------------------


class TestFlatTokenInputToModelInput:
    def test_empty_input(self):
        result = _flat_token_input_to_model_input([])
        assert result.chunks == []

    def test_pure_ints(self):
        result = _flat_token_input_to_model_input([1, 2, 3])
        assert len(result.chunks) == 1
        assert isinstance(result.chunks[0], tinker.EncodedTextChunk)
        assert result.chunks[0].tokens == [1, 2, 3]

    def test_image_chunk_splits_text(self):
        """An image chunk in the middle should produce text-image-text."""
        img = _make_image_chunk(16)
        result = _flat_token_input_to_model_input([1, 2, img, 3, 4])
        assert len(result.chunks) == 3
        assert isinstance(result.chunks[0], tinker.EncodedTextChunk)
        assert result.chunks[0].tokens == [1, 2]
        assert result.chunks[1] is img
        assert isinstance(result.chunks[2], tinker.EncodedTextChunk)
        assert result.chunks[2].tokens == [3, 4]

    def test_leading_image_chunk(self):
        """Image at the start should not produce an empty text chunk."""
        img = _make_image_chunk(8)
        result = _flat_token_input_to_model_input([img, 1, 2])
        assert len(result.chunks) == 2
        assert result.chunks[0] is img
        assert isinstance(result.chunks[1], tinker.EncodedTextChunk)
        assert result.chunks[1].tokens == [1, 2]

    def test_trailing_image_chunk(self):
        img = _make_image_chunk(8)
        result = _flat_token_input_to_model_input([1, 2, img])
        assert len(result.chunks) == 2
        assert isinstance(result.chunks[0], tinker.EncodedTextChunk)
        assert result.chunks[0].tokens == [1, 2]
        assert result.chunks[1] is img

    def test_consecutive_image_chunks(self):
        img1 = _make_image_chunk(8)
        img2 = _make_image_chunk(16)
        result = _flat_token_input_to_model_input([img1, img2])
        assert len(result.chunks) == 2
        assert result.chunks[0] is img1
        assert result.chunks[1] is img2


# ------------------------------------------------------------------
# _flat_token_input_length
# ------------------------------------------------------------------


class TestFlatTokenInputLength:
    def test_empty(self):
        assert _flat_token_input_length([]) == 0

    def test_pure_ints(self):
        assert _flat_token_input_length([1, 2, 3, 4]) == 4

    def test_mixed(self):
        img = _make_image_chunk(16)
        assert _flat_token_input_length([1, 2, img, 3]) == 3 + 16

    def test_only_image_chunks(self):
        img1 = _make_image_chunk(8)
        img2 = _make_image_chunk(12)
        assert _flat_token_input_length([img1, img2]) == 20


# ------------------------------------------------------------------
# TinkerEngine._convert_images_to_content_list (static method)
# ------------------------------------------------------------------


class TestConvertImagesToContentList:
    @staticmethod
    def _call(messages):
        from rllm.engine.rollout.tinker_engine import TinkerEngine

        return TinkerEngine._convert_images_to_content_list(messages)

    def test_no_images_passthrough(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = self._call(msgs)
        assert result == msgs

    def test_empty_images_passthrough(self):
        msgs = [{"role": "user", "content": "hello", "images": []}]
        result = self._call(msgs)
        assert result == msgs

    def test_images_converted_to_content_list(self):
        fake_img = MagicMock()  # simulates a PIL Image
        msgs = [{"role": "user", "content": "describe this", "images": [fake_img]}]
        result = self._call(msgs)

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        # First element should be the image part
        assert content[0] == {"type": "image", "image": fake_img}
        # Second element should be the text part
        assert content[1] == {"type": "text", "text": "describe this"}
        # images key should be removed
        assert "images" not in result[0]

    def test_multiple_images(self):
        img1 = MagicMock()
        img2 = MagicMock()
        msgs = [{"role": "user", "content": "compare", "images": [img1, img2]}]
        result = self._call(msgs)

        content = result[0]["content"]
        assert len(content) == 3  # 2 images + 1 text
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "text"

    def test_mixed_messages(self):
        """Only messages with images are converted; others pass through."""
        fake_img = MagicMock()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "look at this", "images": [fake_img]},
            {"role": "assistant", "content": "I see it."},
        ]
        result = self._call(msgs)

        assert result[0] == msgs[0]  # system unchanged
        assert isinstance(result[1]["content"], list)  # user converted
        assert result[2] == msgs[2]  # assistant unchanged


# ------------------------------------------------------------------
# TinkerEngine._prepare_max_tokens
# ------------------------------------------------------------------


class TestPrepareMaxTokens:
    @staticmethod
    def _make_engine(max_model_length=1000, max_response_length=256):
        """Create a minimal mock TinkerEngine with the fields _prepare_max_tokens needs."""
        from rllm.engine.rollout.tinker_engine import TinkerEngine

        engine = object.__new__(TinkerEngine)
        # _prepare_max_tokens only reads max_model_length (already decremented by 1 in __init__)
        engine.max_model_length = max_model_length - 1
        engine.max_response_length = max_response_length
        return engine

    def test_within_budget(self):
        engine = self._make_engine(max_model_length=1000)
        # Plenty of room: 999 - 100 = 899 remaining > 256
        assert engine._prepare_max_tokens(256, prompt_length=100) == 256

    def test_capped_by_model_length(self):
        engine = self._make_engine(max_model_length=500)
        # 499 - 400 = 99 remaining < 256
        assert engine._prepare_max_tokens(256, prompt_length=400) == 99

    def test_exact_boundary(self):
        engine = self._make_engine(max_model_length=500)
        # 499 - 243 = 256 remaining == 256
        assert engine._prepare_max_tokens(256, prompt_length=243) == 256

    def test_no_model_length_constraint(self):
        """When max_model_length is 0 (falsy), no capping occurs."""
        engine = object.__new__(type("E", (), {}))
        from rllm.engine.rollout.tinker_engine import TinkerEngine

        engine = object.__new__(TinkerEngine)
        engine.max_model_length = 0  # falsy → skip capping
        engine.max_response_length = 256
        assert engine._prepare_max_tokens(512, prompt_length=100) == 512
