"""Tests for VLM utility functions (vlm_utils.py)."""

from __future__ import annotations

import base64

import pytest


@pytest.fixture
def image_dir(tmp_path, monkeypatch):
    """Create a temporary dataset directory with a test image."""
    datasets_dir = tmp_path / "datasets"
    img_dir = datasets_dir / "test_bench" / "images"
    img_dir.mkdir(parents=True)

    png_bytes = _make_png_bytes()
    (img_dir / "test_0_image.png").write_bytes(png_bytes)
    (img_dir / "test_1_image.png").write_bytes(png_bytes)

    monkeypatch.setattr(
        "rllm.experimental.agents.vlm_utils._DATASETS_ROOT",
        str(datasets_dir),
    )

    return str(datasets_dir)


def _make_png_bytes():
    """Create a minimal 1x1 PNG."""
    import struct
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw_data = zlib.compress(b"\x00\xff\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + raw_data) & 0xFFFFFFFF
    idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


def _make_jpeg_bytes():
    """Create minimal JPEG-like bytes (just the header for detection)."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 20


def _make_webp_bytes():
    """Create minimal WebP-like bytes (just the header for detection)."""
    return b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20


# ---------------------------------------------------------------------------
# _detect_mime_type
# ---------------------------------------------------------------------------


class TestDetectMimeType:
    def test_png(self):
        from rllm.experimental.agents.vlm_utils import _detect_mime_type

        assert _detect_mime_type(_make_png_bytes()) == "image/png"

    def test_jpeg(self):
        from rllm.experimental.agents.vlm_utils import _detect_mime_type

        assert _detect_mime_type(_make_jpeg_bytes()) == "image/jpeg"

    def test_webp(self):
        from rllm.experimental.agents.vlm_utils import _detect_mime_type

        assert _detect_mime_type(_make_webp_bytes()) == "image/webp"

    def test_unknown_defaults_to_png(self):
        from rllm.experimental.agents.vlm_utils import _detect_mime_type

        assert _detect_mime_type(b"\x00\x00\x00\x00") == "image/png"


# ---------------------------------------------------------------------------
# _image_to_data_uri
# ---------------------------------------------------------------------------


class TestImageToDataUri:
    def test_bytes_png(self):
        from rllm.experimental.agents.vlm_utils import _image_to_data_uri

        png = _make_png_bytes()
        uri = _image_to_data_uri(png)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",", 1)[1])
        assert decoded == png

    def test_bytes_jpeg(self):
        from rllm.experimental.agents.vlm_utils import _image_to_data_uri

        jpeg = _make_jpeg_bytes()
        uri = _image_to_data_uri(jpeg)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_str_path(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _image_to_data_uri

        uri = _image_to_data_uri("test_bench/images/test_0_image.png")
        assert uri.startswith("data:image/png;base64,")

    def test_invalid_type_raises(self):
        from rllm.experimental.agents.vlm_utils import _image_to_data_uri

        with pytest.raises(TypeError):
            _image_to_data_uri(12345)


# ---------------------------------------------------------------------------
# _load_image_as_data_uri
# ---------------------------------------------------------------------------


class TestLoadImageAsDataURI:
    def test_loads_image(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _load_image_as_data_uri

        uri = _load_image_as_data_uri("test_bench/images/test_0_image.png")
        assert uri.startswith("data:image/png;base64,")
        b64_part = uri.split(",", 1)[1]
        data = base64.b64decode(b64_part)
        assert data[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# _build_vlm_content
# ---------------------------------------------------------------------------


class TestBuildVLMContent:
    def test_single_image(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        content = _build_vlm_content("hello", ["test_bench/images/test_0_image.png"])
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "hello"

    def test_multiple_images(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        content = _build_vlm_content(
            "question",
            [
                "test_bench/images/test_0_image.png",
                "test_bench/images/test_1_image.png",
            ],
        )
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_no_images(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        content = _build_vlm_content("text only", [])
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_missing_image_skipped(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        content = _build_vlm_content("text", ["nonexistent/path.png"])
        assert len(content) == 1
        assert content[0]["type"] == "text"

    def test_single_bytes_image(self):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        png = _make_png_bytes()
        content = _build_vlm_content("hello", [png])
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "hello"

    def test_multiple_bytes_images(self):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        png = _make_png_bytes()
        jpeg = _make_jpeg_bytes()
        content = _build_vlm_content("q", [png, jpeg])
        assert len(content) == 3
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_mixed_bytes_and_paths(self, image_dir):
        from rllm.experimental.agents.vlm_utils import _build_vlm_content

        png = _make_png_bytes()
        content = _build_vlm_content("q", [png, "test_bench/images/test_0_image.png"])
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
