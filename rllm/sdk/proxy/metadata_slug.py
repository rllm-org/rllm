"""Metadata slug helpers shared between the SDK and proxy."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

from rllm.sdk.session import (
    get_active_session_uids,
    get_current_metadata,
    get_current_session_name,
)

_SLUG_PREFIX = "rllm1:"


def assemble_routing_metadata(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the metadata dict that should be routed through the proxy slug."""
    payload: dict[str, Any] = dict(get_current_metadata())
    session_name = get_current_session_name()
    if session_name and "session_name" not in payload:
        payload["session_name"] = session_name

    # Add session UIDs from active context (backend-agnostic)
    uids = get_active_session_uids()
    if uids:
        payload["session_uids"] = uids

    if extra:
        payload.update(dict(extra))
    return payload


def encode_metadata_slug(metadata: Mapping[str, Any]) -> str:
    """Encode metadata into a versioned slug suitable for URL embedding."""
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
    return f"{_SLUG_PREFIX}{encoded.decode('ascii')}"


def decode_metadata_slug(slug: str) -> dict[str, Any]:
    """Decode a versioned slug back into the original metadata dict."""
    if not slug.startswith(_SLUG_PREFIX):
        raise ValueError(f"Unsupported metadata slug: {slug}")
    payload = slug[len(_SLUG_PREFIX) :]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
    return json.loads(decoded)


def build_proxied_base_url(base_url: str, metadata: Mapping[str, Any]) -> str:
    """Return a base URL augmented with the metadata slug."""
    slug = encode_metadata_slug(metadata)
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    has_v1 = path.endswith("/v1")
    if has_v1:
        path = path[:-3]
    new_path = f"{path}/meta/{slug}"
    if has_v1:
        new_path += "/v1"
    if not new_path.startswith("/"):
        new_path = "/" + new_path
    rebuilt = parsed._replace(path=new_path)
    return urlunparse(rebuilt)


def extract_metadata_from_path(path: str) -> tuple[str, dict[str, Any]] | None:
    """Extract the metadata slug from the request path and return the clean path and dict."""
    segments = path.split("/")
    for idx, segment in enumerate(segments):
        if segment == "meta" and idx + 1 < len(segments):
            slug = segments[idx + 1]
            metadata = decode_metadata_slug(slug)
            cleaned_segments = segments[:idx] + segments[idx + 2 :]
            cleaned_path = "/".join(cleaned_segments)
            if cleaned_path == "":
                cleaned_path = "/"
            elif not cleaned_path.startswith("/"):
                cleaned_path = "/" + cleaned_path
            return cleaned_path, metadata
    return None
