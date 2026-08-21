"""Audio/video grading for GDPval: ffprobe metadata + a direct-Gemini judge.

Two of GDPval's 220 tasks expect a video deliverable, and a text+image judge
cannot watch one. They also cannot ride the normal chat lane: there is no files
API there, and inlining base64 blows the request-size limit on real files (the
dataset's media runs from single-digit MB into the hundreds). So they go through
Google's Files API instead::

    upload bytes -> file name/uri -> poll until ACTIVE -> generateContent with a
    file_data part -> the judge natively watches/listens

This needs ``GEMINI_API_KEY`` — a *direct* Google AI Studio key. Proxy or
aggregator keys do not work: the Files API is Google-specific, and an uploaded
file only resolves for the key that uploaded it. Without one, media tasks come
back ungraded rather than scored from a filename.

:func:`ffprobe_metadata` supplies the technical facts (container, duration,
codec, resolution, fps) as authoritative text alongside the file, so a criterion
like "a 1080p MP4 under 60 seconds" is answered from the container rather than
from whatever the solver claimed. ffprobe always runs inside the published
GDPval image, which carries ffmpeg as part of AA's package closure, so the
metadata does not vary with the judge host.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".aac", ".ogg", ".aiff", ".m4a", ".opus"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".webm", ".wmv", ".mkv", ".mpg", ".mpeg"})
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

_GEMINI_BASE = "https://generativelanguage.googleapis.com"


def is_media(path: str | Path) -> bool:
    """True for audio/video, which only the media judge can perceive."""
    return Path(path).suffix.lower() in MEDIA_EXTS


def media_api_key() -> str | None:
    """The direct Google key enabling native media grading (None = disabled)."""
    return os.environ.get("GEMINI_API_KEY") or None


def media_judge_model() -> str:
    """Gemini handles audio/video natively; AA routes media comparisons to it too."""
    return os.environ.get("GDPVAL_MEDIA_JUDGE_MODEL", "gemini-3.1-pro-preview")


def _max_media_bytes() -> int:
    """Files-API per-file ceiling (Google caps at 2 GB; default guard 1900 MB)."""
    return int(os.environ.get("GDPVAL_MEDIA_MAX_MB", "1900")) * 1024 * 1024


def media_mime(path: str | Path) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    ext = Path(path).suffix.lower().lstrip(".")
    return f"audio/{ext}" if f".{ext}" in AUDIO_EXTS else f"video/{ext}"


def render_docker_image() -> str:
    """Image carrying LibreOffice and ffmpeg for host-independent conversion.

    Defaults to the published GDPval sandbox image: AA's closure includes both,
    so a host with neither installed still renders page images and reads media
    metadata instead of silently going without.
    """
    from rllm.data import gdpval_aa

    fallback = gdpval_aa.published_image_ref() or "gdpval-sandbox"
    return os.environ.get("GDPVAL_RENDER_DOCKER_IMAGE", fallback)


# --------------------------------------------------------------------------- #
# ffprobe technical metadata (pinned docker image)
# --------------------------------------------------------------------------- #


def _ffprobe_json(path: Path) -> dict | None:
    """Parsed ffprobe JSON, or None.

    Runs ffprobe from the pinned GDPval image only, for the same reason
    :func:`rllm.eval.reward_fns.gdpval_rubric._soffice_convert` does: the
    strings it returns are what spec-shaped criteria ("a 1080p MP4 under 60
    seconds") are graded against, and a host binary of another build names the
    same codec and container differently. A host fallback would make the score
    depend on which machine ran the judge.
    """
    import subprocess

    from rllm.data.gdpval_aa import AA_PLATFORM

    src = path.resolve()
    try:
        out = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                AA_PLATFORM,
                "--network",
                "none",
                "--mount",
                f"type=bind,source={src.parent},target=/input,readonly",
                render_docker_image(),
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                f"/input/{src.name}",
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return json.loads(out.stdout.decode("utf-8", errors="replace"))
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.warning("[gdpval] Docker ffprobe failed for %s: %s", path.name, e)
        return None


def ffprobe_metadata(path: str | Path) -> str:
    """Technical metadata as text — authoritative for spec-shaped criteria."""
    p = Path(path)
    data = _ffprobe_json(p)
    if not data:
        return f"# Media file: {p.name}\n[technical metadata unavailable]"
    lines = [f"# Media file: {p.name}", "Technical metadata (ffprobe):"]
    fmt = data.get("format", {})
    for key, label in (("format_long_name", "container"), ("duration", "duration_s"), ("size", "size_bytes"), ("bit_rate", "bit_rate")):
        if fmt.get(key):
            lines.append(f"- {label}: {fmt[key]}")
    for stream in data.get("streams", []):
        kind = stream.get("codec_type", "stream")
        desc = [stream.get("codec_name", "?")]
        if kind == "video":
            desc.append(f"{stream.get('width', '?')}x{stream.get('height', '?')}")
            if stream.get("r_frame_rate") and stream["r_frame_rate"] != "0/0":
                desc.append(f"{stream['r_frame_rate']} fps")
        if kind == "audio":
            desc.append(f"{stream.get('sample_rate', '?')} Hz")
            desc.append(f"{stream.get('channels', '?')} ch")
        lines.append(f"- {kind} stream: " + ", ".join(str(d) for d in desc))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gemini Files API upload (bytes -> file_uri, polled until ACTIVE)
# --------------------------------------------------------------------------- #


class MediaUploadError(RuntimeError):
    """Upload/processing failed — the caller must NOT grade the media task."""


def upload_media(path: str | Path, api_key: str, *, timeout_s: float = 300.0, poll_s: float = 3.0) -> dict:
    """Upload one media file; return ``{"file_uri", "mime_type", "name"}``.

    Raises :class:`MediaUploadError` on any failure — oversize, HTTP error,
    processing FAILED, or still not ACTIVE at the deadline — so callers fail
    closed instead of grading a file the judge never received.
    """
    import httpx

    p = Path(path)
    size = p.stat().st_size
    if size > _max_media_bytes():
        raise MediaUploadError(f"{p.name} is {size / 1e6:.0f} MB, over the {_max_media_bytes() / 1e6:.0f} MB media limit")
    mime = media_mime(p)

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            # 1. initiate a resumable upload -> one-time upload URL
            start = client.post(
                f"{_GEMINI_BASE}/upload/v1beta/files",
                params={"key": api_key},
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": mime,
                    "Content-Type": "application/json",
                },
                json={"file": {"display_name": p.name}},
            )
            start.raise_for_status()
            upload_url = start.headers.get("x-goog-upload-url")
            if not upload_url:
                raise MediaUploadError("no upload URL returned")

            # 2. transfer the bytes (single shot; Gemini accepts up to 2 GB)
            up = client.post(
                upload_url,
                headers={"X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0", "Content-Length": str(size)},
                content=p.read_bytes(),
            )
            up.raise_for_status()
            info = up.json().get("file", {})
            name, uri = info.get("name"), info.get("uri")
            if not (name and uri):
                raise MediaUploadError(f"malformed upload response: {up.text[:200]}")

            # 3. poll until server-side ingestion finishes; a file is only
            #    referenceable once it reaches ACTIVE.
            deadline = time.monotonic() + timeout_s
            state = info.get("state")
            while state == "PROCESSING" or state is None:
                if time.monotonic() > deadline:
                    raise MediaUploadError(f"{p.name} not ACTIVE after {timeout_s:.0f}s")
                time.sleep(poll_s)
                got = client.get(f"{_GEMINI_BASE}/v1beta/{name}", params={"key": api_key})
                got.raise_for_status()
                state = got.json().get("state")
            if state != "ACTIVE":
                raise MediaUploadError(f"{p.name} ingestion ended in state {state}")
            return {"file_uri": uri, "mime_type": mime, "name": name}
    except MediaUploadError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize transport errors
        raise MediaUploadError(f"upload failed for {p.name}: {e}") from e


# --------------------------------------------------------------------------- #
# The media judge: OpenAI-style messages -> Gemini generateContent (+ files)
# --------------------------------------------------------------------------- #


def _to_gemini_contents(messages: list[dict], media_files: list[dict]) -> tuple[dict | None, list[dict]]:
    """Convert OpenAI-style messages to ``(system_instruction, contents)``.

    Text parts pass through; base64 ``image_url`` parts become ``inline_data``;
    the uploaded media files are appended as ``file_data`` parts on the first
    user message, so the judge sees them alongside the task text.
    """
    system_instruction = None
    contents: list[dict] = []
    files_pending = list(media_files)
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts: list[dict] = []
        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for part in content:
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:") and ";base64," in url:
                        mime, b64 = url[5:].split(";base64,", 1)
                        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        if role == "system":
            system_instruction = {"parts": parts}
            continue
        if role == "user" and files_pending:
            parts.extend({"file_data": {"file_uri": f["file_uri"], "mime_type": f["mime_type"]}} for f in files_pending)
            files_pending = []
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return system_instruction, contents


def make_media_judge(model: str, api_key: str, media_files: list[dict], *, max_tokens: int = 65536):
    """Return ``call(messages, text_only_messages=None) -> text``.

    Same signature as the litellm rubric judge, so the chunking, retry and
    verification machinery runs unchanged on media tasks. Calls Gemini's
    ``generateContent`` directly rather than going through the chat lane.
    """
    import httpx

    def call(messages: list[dict], text_only_messages: list[dict] | None = None) -> str:  # noqa: ARG001 — no text fallback: the media *is* the evidence
        system_instruction, contents = _to_gemini_contents(messages, media_files)
        body: dict = {"contents": contents, "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens}}
        if system_instruction:
            body["systemInstruction"] = system_instruction
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            resp = client.post(f"{_GEMINI_BASE}/v1beta/models/{model}:generateContent", params={"key": api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    return call
